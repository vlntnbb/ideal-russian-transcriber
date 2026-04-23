import AppKit
import Foundation
import SwiftUI
import UniformTypeIdentifiers

enum BatchMode: String, CaseIterable, Identifiable {
    case separate
    case merge

    var id: String { rawValue }

    var title: String {
        switch self {
        case .separate:
            return "По отдельности"
        case .merge:
            return "Объединить в одно"
        }
    }
}

struct BatchRow: Decodable, Identifiable {
    let status: String?
    let input_path: String?
    let output_dir: String?
    let transcript_md: String?
    let result_json: String?
    let error: String?
    let source_inputs: [String]?

    var id: String {
        "\(output_dir ?? "")::\(input_path ?? UUID().uuidString)"
    }
}

struct BatchSummary: Decodable {
    let status: String
    let mode: String?
    let started_at: String?
    let finished_at: String?
    let output_root: String?
    let normalize_audio: Bool?
    let dry_run: Bool?
    let inputs: [String]?
    let results: [BatchRow]
    let errors: [String]?
}

@MainActor
final class AppViewModel: ObservableObject {
    @Published var projectRoot: String
    @Published var selectedFiles: [URL] = []
    @Published var mode: BatchMode = .separate
    @Published var normalizeAudio: Bool = false
    @Published var dryRun: Bool = false
    @Published var isRunning: Bool = false
    @Published var logs: String = ""
    @Published var summary: BatchSummary?
    @Published var errorText: String?
    @Published var lastOutputRoot: String = ""

    private var activeProcess: Process?

    init() {
        self.projectRoot = Self.detectProjectRoot()
    }

    static func detectProjectRoot() -> String {
        let fm = FileManager.default
        if let explicit = ProcessInfo.processInfo.environment["TRANSCRIBE_PROJECT_ROOT"], !explicit.isEmpty {
            return explicit
        }

        var url = URL(fileURLWithPath: fm.currentDirectoryPath, isDirectory: true)
        for _ in 0..<8 {
            let candidate = url.appendingPathComponent("transcribe_batch_cli.py")
            if fm.fileExists(atPath: candidate.path) {
                return url.path
            }
            let parent = url.deletingLastPathComponent()
            if parent.path == url.path {
                break
            }
            url = parent
        }
        return fm.currentDirectoryPath
    }

    func pickProjectRoot() {
        let panel = NSOpenPanel()
        panel.canChooseFiles = false
        panel.canChooseDirectories = true
        panel.allowsMultipleSelection = false
        panel.prompt = "Выбрать"
        panel.message = "Выберите корень проекта (где лежит transcribe_batch_cli.py)"
        if panel.runModal() == .OK, let url = panel.url {
            projectRoot = url.path
        }
    }

    func pickFiles() {
        let panel = NSOpenPanel()
        panel.canChooseFiles = true
        panel.canChooseDirectories = false
        panel.allowsMultipleSelection = true
        panel.allowedContentTypes = [.audio, .movie]
        panel.prompt = "Добавить"
        panel.message = "Выберите аудио/видео файлы для транскрибации"
        if panel.runModal() == .OK {
            selectedFiles = panel.urls.sorted { $0.lastPathComponent.localizedCompare($1.lastPathComponent) == .orderedAscending }
            if selectedFiles.count <= 1 {
                mode = .separate
            }
        }
    }

    func clearSelection() {
        selectedFiles = []
        summary = nil
        errorText = nil
    }

    func cancelRun() {
        activeProcess?.terminate()
    }

    func openPath(_ path: String?) {
        guard let path, !path.isEmpty else { return }
        NSWorkspace.shared.open(URL(fileURLWithPath: path))
    }

    func startBatch() {
        Task {
            await runBatch()
        }
    }

    private func appendLog(_ line: String) {
        if line.isEmpty { return }
        logs.append(line)
        if !line.hasSuffix("\n") {
            logs.append("\n")
        }
    }

    private func makeOutputRootURL() -> URL {
        let ts = ISO8601DateFormatter().string(from: Date()).replacingOccurrences(of: ":", with: "-")
        let root = URL(fileURLWithPath: projectRoot, isDirectory: true)
            .appendingPathComponent("swift_app_runs", isDirectory: true)
            .appendingPathComponent(ts, isDirectory: true)
        return root
    }

    private func pythonInvocation(scriptPath: String) -> (executable: URL, args: [String]) {
        let venvPython = URL(fileURLWithPath: projectRoot, isDirectory: true).appendingPathComponent(".venv/bin/python3")
        if FileManager.default.isExecutableFile(atPath: venvPython.path) {
            return (venvPython, [scriptPath])
        }
        return (URL(fileURLWithPath: "/usr/bin/env"), ["python3", scriptPath])
    }

    private func runBatch() async {
        guard !selectedFiles.isEmpty else {
            errorText = "Сначала выберите хотя бы один файл."
            return
        }

        let rootURL = URL(fileURLWithPath: projectRoot, isDirectory: true)
        let scriptURL = rootURL.appendingPathComponent("transcribe_batch_cli.py")
        guard FileManager.default.fileExists(atPath: scriptURL.path) else {
            errorText = "Не найден transcribe_batch_cli.py в \(projectRoot)"
            return
        }

        errorText = nil
        summary = nil
        logs = ""
        isRunning = true

        let outputRoot = makeOutputRootURL()
        do {
            try FileManager.default.createDirectory(at: outputRoot, withIntermediateDirectories: true)
        } catch {
            isRunning = false
            errorText = "Не удалось создать папку результатов: \(error.localizedDescription)"
            return
        }
        lastOutputRoot = outputRoot.path

        let selectedMode: BatchMode = (selectedFiles.count > 1) ? mode : .separate
        let summaryName = "summary.json"
        var arguments: [String] = [
            "--mode", selectedMode.rawValue,
            "--out", outputRoot.path,
            "--summary-file", summaryName,
        ]
        if normalizeAudio {
            arguments.append("--norm")
        }
        if dryRun {
            arguments.append("--dry-run")
        }
        arguments.append("--inputs")
        arguments.append(contentsOf: selectedFiles.map(\.path))

        let invocation = pythonInvocation(scriptPath: scriptURL.path)
        let cmdPreview = ([invocation.executable.path] + invocation.args + arguments).joined(separator: " ")
        appendLog("Запуск: \(cmdPreview)")

        do {
            let status = try await runProcess(
                executable: invocation.executable,
                args: invocation.args + arguments,
                currentDirectory: rootURL
            )
            appendLog("Процесс завершён, код выхода: \(status)")
        } catch {
            appendLog("Ошибка запуска: \(error.localizedDescription)")
            errorText = "Не удалось запустить batch-CLI: \(error.localizedDescription)"
            isRunning = false
            return
        }

        let summaryURL = outputRoot.appendingPathComponent(summaryName)
        do {
            let data = try Data(contentsOf: summaryURL)
            let decoded = try JSONDecoder().decode(BatchSummary.self, from: data)
            summary = decoded
            if decoded.status != "ok" {
                errorText = "Выполнение завершилось с ошибками."
            }
        } catch {
            errorText = "Не удалось прочитать summary.json: \(error.localizedDescription)"
        }

        isRunning = false
    }

    private func runProcess(executable: URL, args: [String], currentDirectory: URL) async throws -> Int32 {
        let process = Process()
        process.executableURL = executable
        process.arguments = args
        process.currentDirectoryURL = currentDirectory

        let stdout = Pipe()
        let stderr = Pipe()
        process.standardOutput = stdout
        process.standardError = stderr

        func attachReader(_ fh: FileHandle) {
            fh.readabilityHandler = { [weak self] handle in
                let data = handle.availableData
                if data.isEmpty { return }
                let text = String(data: data, encoding: .utf8) ?? "<non-utf8 chunk>"
                Task { @MainActor in
                    self?.appendLog(text)
                }
            }
        }
        attachReader(stdout.fileHandleForReading)
        attachReader(stderr.fileHandleForReading)

        try process.run()
        activeProcess = process

        let status: Int32 = await withCheckedContinuation { continuation in
            process.terminationHandler = { proc in
                continuation.resume(returning: proc.terminationStatus)
            }
        }

        stdout.fileHandleForReading.readabilityHandler = nil
        stderr.fileHandleForReading.readabilityHandler = nil
        activeProcess = nil
        return status
    }
}

struct ContentView: View {
    @StateObject private var vm = AppViewModel()

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Ideal Russian Transcribe (macOS)")
                .font(.title2.bold())

            HStack {
                Text("Проект:")
                TextField("Путь к проекту", text: $vm.projectRoot)
                    .textFieldStyle(.roundedBorder)
                Button("Выбрать папку…") {
                    vm.pickProjectRoot()
                }
            }

            HStack {
                Button("Выбрать аудио/видео…") {
                    vm.pickFiles()
                }
                Button("Очистить") {
                    vm.clearSelection()
                }
                Text("Файлов: \(vm.selectedFiles.count)")
                    .foregroundStyle(.secondary)
            }

            if vm.selectedFiles.count > 1 {
                Picker("Режим обработки:", selection: $vm.mode) {
                    ForEach(BatchMode.allCases) { mode in
                        Text(mode.title).tag(mode)
                    }
                }
                .pickerStyle(.segmented)
            }

            HStack {
                Toggle("Нормализация аудио", isOn: $vm.normalizeAudio)
                Toggle("Dry run", isOn: $vm.dryRun)
            }

            HStack {
                Button("Запустить") {
                    vm.startBatch()
                }
                .disabled(vm.selectedFiles.isEmpty || vm.isRunning)

                Button("Остановить") {
                    vm.cancelRun()
                }
                .disabled(!vm.isRunning)

                if vm.isRunning {
                    ProgressView()
                }
            }

            if !vm.selectedFiles.isEmpty {
                List(vm.selectedFiles, id: \.path) { url in
                    Text(url.path)
                        .font(.system(.body, design: .monospaced))
                        .lineLimit(1)
                }
                .frame(minHeight: 140, maxHeight: 200)
            }

            if let summary = vm.summary {
                VStack(alignment: .leading, spacing: 8) {
                    Text("Результат: \(summary.status.uppercased())")
                        .font(.headline)
                    if let root = summary.output_root {
                        HStack {
                            Text("Папка вывода: \(root)")
                                .font(.system(.caption, design: .monospaced))
                                .lineLimit(1)
                            Button("Открыть") {
                                vm.openPath(root)
                            }
                        }
                    }
                    List(summary.results) { row in
                        VStack(alignment: .leading, spacing: 6) {
                            HStack {
                                Text(row.status == "ok" ? "OK" : "ERROR")
                                    .font(.caption.bold())
                                    .foregroundStyle(row.status == "ok" ? .green : .red)
                                Text(row.input_path ?? "")
                                    .font(.system(.caption, design: .monospaced))
                                    .lineLimit(1)
                                Spacer()
                                if let transcript = row.transcript_md {
                                    Button("Transcript") {
                                        vm.openPath(transcript)
                                    }
                                }
                                if let out = row.output_dir {
                                    Button("Папка") {
                                        vm.openPath(out)
                                    }
                                }
                            }
                            if let err = row.error, !err.isEmpty {
                                Text(err)
                                    .foregroundStyle(.red)
                                    .font(.caption)
                                    .lineLimit(2)
                            }
                        }
                    }
                    .frame(minHeight: 120, maxHeight: 180)
                }
            }

            Text("Лог")
                .font(.headline)
            TextEditor(text: $vm.logs)
                .font(.system(.caption, design: .monospaced))
                .frame(minHeight: 220)

            if let error = vm.errorText, !error.isEmpty {
                Text(error)
                    .foregroundStyle(.red)
            }
        }
        .padding(16)
        .frame(minWidth: 980, minHeight: 760)
    }
}

@main
struct IdealRussianTranscribeApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
        }
        .windowResizability(.contentSize)
    }
}
