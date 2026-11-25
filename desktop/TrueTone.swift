import Cocoa
import Darwin
import WebKit
import AVFoundation
import UniformTypeIdentifiers

final class AppDelegate: NSObject, NSApplicationDelegate, WKUIDelegate, WKNavigationDelegate {
    var window: NSWindow!
    var web: WKWebView!
    var loading: NSView!
    var status: NSTextField!
    var timer: Timer?
    var server: Process?
    var started = Date()
    var checking = false
    var finished = false
    var recoveredWebProcess = false
    let address = URL(string: "http://127.0.0.1:7861/?__theme=light")!
    let log = FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent("Library/Logs/TrueTone/server.log")

    func applicationDidFinishLaunching(_ notification: Notification) {
        let menu = NSMenu()
        let item = NSMenuItem()
        menu.addItem(item)
        let appMenu = NSMenu()
        appMenu.addItem(withTitle: "Enable Microphone", action: #selector(enableMicrophone), keyEquivalent: "m")
        appMenu.addItem(withTitle: "Grant Project Folder Access", action: #selector(grantProjectAccess), keyEquivalent: "")
        appMenu.addItem(withTitle: "Open in Browser", action: #selector(openBrowser), keyEquivalent: "o")
        appMenu.addItem(withTitle: "Reload Workspace", action: #selector(reloadWorkspace), keyEquivalent: "r")
        appMenu.addItem(withTitle: "Close Window", action: #selector(closeWindow), keyEquivalent: "w")
        appMenu.addItem(withTitle: "Quit TrueTone", action: #selector(NSApplication.terminate(_:)), keyEquivalent: "q")
        item.submenu = appMenu
        let editItem = NSMenuItem(title: "Edit", action: nil, keyEquivalent: "")
        let editMenu = NSMenu(title: "Edit")
        for (title, action, key) in [("Cut", #selector(NSText.cut(_:)), "x"),
                                     ("Copy", #selector(NSText.copy(_:)), "c"),
                                     ("Paste", #selector(NSText.paste(_:)), "v"),
                                     ("Select All", #selector(NSText.selectAll(_:)), "a")] {
            editMenu.addItem(withTitle: title, action: action, keyEquivalent: key)
        }
        editItem.submenu = editMenu
        menu.addItem(editItem)
        NSApp.mainMenu = menu
        window = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 1120, height: 850),
                          styleMask: [.titled, .closable, .miniaturizable, .resizable], backing: .buffered, defer: false)
        window.title = "TrueTone"
        window.minSize = NSSize(width: 390, height: 600)
        window.center()
        let configuration = WKWebViewConfiguration()
        configuration.websiteDataStore = .nonPersistent()
        web = WKWebView(frame: window.contentView!.bounds, configuration: configuration)
        web.autoresizingMask = [.width, .height]
        web.uiDelegate = self
        web.navigationDelegate = self
        window.contentView!.addSubview(web)
        loading = NSView(frame: window.contentView!.bounds)
        loading.autoresizingMask = [.width, .height]
        loading.wantsLayer = true
        loading.layer?.backgroundColor = NSColor(calibratedRed: 0.96, green: 0.95, blue: 0.91, alpha: 1).cgColor
        window.contentView!.addSubview(loading)
        let title = NSTextField(labelWithString: "TrueTone")
        title.font = NSFont(name: "Georgia", size: 48)
        title.textColor = NSColor(calibratedRed: 0.13, green: 0.24, blue: 0.20, alpha: 1)
        title.translatesAutoresizingMaskIntoConstraints = false
        status = NSTextField(wrappingLabelWithString: "Starting your local workspace · 0s")
        status.alignment = .center
        status.font = .systemFont(ofSize: 16)
        status.textColor = .darkGray
        status.translatesAutoresizingMaskIntoConstraints = false
        let logo = NSImageView()
        logo.image = NSImage(contentsOf: Bundle.main.url(forResource: "logo", withExtension: "png")!)
        logo.imageScaling = .scaleProportionallyUpOrDown
        logo.translatesAutoresizingMaskIntoConstraints = false
        loading.addSubview(logo)
        loading.addSubview(title)
        loading.addSubview(status)
        NSLayoutConstraint.activate([
            logo.widthAnchor.constraint(equalToConstant: 88),
            logo.heightAnchor.constraint(equalToConstant: 88),
            logo.centerXAnchor.constraint(equalTo: loading.centerXAnchor),
            logo.bottomAnchor.constraint(equalTo: title.topAnchor, constant: -24),
            title.centerXAnchor.constraint(equalTo: loading.centerXAnchor),
            title.centerYAnchor.constraint(equalTo: loading.centerYAnchor, constant: -45),
            status.centerXAnchor.constraint(equalTo: loading.centerXAnchor),
            status.topAnchor.constraint(equalTo: title.bottomAnchor, constant: 24),
            status.widthAnchor.constraint(lessThanOrEqualTo: loading.widthAnchor, constant: -48)
        ])
        window.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
        timer = Timer.scheduledTimer(withTimeInterval: 0.5, repeats: true) { [weak self] _ in self?.poll() }
        startServer()
        poll()
    }

    func startServer() {
        do {
            let pathFile = Bundle.main.url(forResource: "project-path", withExtension: "txt")!
            let root = try String(contentsOf: pathFile, encoding: .utf8).trimmingCharacters(in: .whitespacesAndNewlines)
            let python = URL(fileURLWithPath: root).appendingPathComponent(".venv/bin/python")
            guard FileManager.default.isExecutableFile(atPath: python.path),
                  FileManager.default.fileExists(atPath: root + "/app.py") else {
                fail("Project or Python environment missing. Rebuild the Desktop app from the project folder.")
                return
            }
            try FileManager.default.createDirectory(at: log.deletingLastPathComponent(), withIntermediateDirectories: true)
            FileManager.default.createFile(atPath: log.path, contents: nil)
            let output = try FileHandle(forWritingTo: log)
            try output.seekToEnd()
            let process = Process()
            process.executableURL = python
            process.arguments = [root + "/scripts/run_desktop_server.py", String(ProcessInfo.processInfo.processIdentifier)]
            process.currentDirectoryURL = URL(fileURLWithPath: root)
            var env = ProcessInfo.processInfo.environment
            env["PYTHONUNBUFFERED"] = "1"
            env["TRUETONE_PORT"] = "7861"
            process.environment = env
            process.standardOutput = output
            process.standardError = output
            try process.run()
            server = process
        } catch { fail("Could not start TrueTone: \(error.localizedDescription)") }
    }

    func poll() {
        guard !finished else { return }
        let seconds = Int(Date().timeIntervalSince(started))
        status.stringValue = "Starting your local workspace · \(seconds)s\nThe website opens automatically when ready."
        if seconds >= 120 { fail("Startup timed out. Check \(log.path), then reopen TrueTone."); return }
        if let process = server, !process.isRunning {
            fail("Server stopped during startup. Check \(log.path), then reopen TrueTone.")
            return
        }
        guard !checking else { return }
        checking = true
        var request = URLRequest(url: URL(string: "http://127.0.0.1:7861/config")!)
        request.timeoutInterval = 2
        URLSession.shared.dataTask(with: request) { [weak self] data, response, error in
            DispatchQueue.main.async {
                guard let self = self, !self.finished else { return }
                self.checking = false
                if let response = response as? HTTPURLResponse {
                    let config = data.flatMap { try? JSONSerialization.jsonObject(with: $0) as? [String: Any] }
                    guard response.statusCode == 200,
                          (config?["title"] as? String)?.hasPrefix("TrueTone") == true else {
                        self.fail("Port 7861 is used by another service. Close that service, then reopen TrueTone.")
                        return
                    }
                    self.finished = true
                    self.timer?.invalidate()
                    self.status.stringValue = "Loading your workspace · \(seconds)s"
                    self.web.load(URLRequest(url: self.address))
                }
            }
        }.resume()
    }

    func fail(_ message: String) {
        finished = true
        timer?.invalidate()
        loading.isHidden = false
        status.stringValue = message
    }
    func webView(_ webView: WKWebView, didFinish navigation: WKNavigation!) { waitForWorkspace() }
    func waitForWorkspace(_ attempt: Int = 0) {
        web.evaluateJavaScript("document.getElementById('workspace') !== null") { [weak self] value, _ in
            guard let self = self else { return }
            if value as? Bool == true {
                self.loading.isHidden = true
                self.window.makeKeyAndOrderFront(nil)
                NSApp.activate(ignoringOtherApps: true)
            }
            else if attempt < 20 {
                DispatchQueue.main.asyncAfter(deadline: .now() + 0.25) { self.waitForWorkspace(attempt + 1) }
            } else if !self.recoveredWebProcess {
                self.recoveredWebProcess = true
                self.reloadWorkspace()
            } else { self.fail("The page did not render. Try Reload Workspace or Open in Browser.") }
        }
    }
    func webView(_ webView: WKWebView, didFailProvisionalNavigation navigation: WKNavigation!, withError error: Error) {
        fail("Website could not load: \(error.localizedDescription). Try Open in Browser from the TrueTone menu.")
    }
    func webView(_ webView: WKWebView, runOpenPanelWith parameters: WKOpenPanelParameters,
                 initiatedByFrame frame: WKFrameInfo, completionHandler: @escaping ([URL]?) -> Void) {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.audio]
        panel.allowsMultipleSelection = parameters.allowsMultipleSelection
        panel.canChooseDirectories = false
        panel.beginSheetModal(for: window) { response in
            completionHandler(response == .OK ? panel.urls : nil)
        }
    }
    func webView(_ webView: WKWebView, createWebViewWith configuration: WKWebViewConfiguration,
                 for navigationAction: WKNavigationAction, windowFeatures: WKWindowFeatures) -> WKWebView? {
        if navigationAction.targetFrame == nil, let url = navigationAction.request.url,
           ["https", "http"].contains(url.scheme ?? "") { NSWorkspace.shared.open(url) }
        return nil
    }
    @objc func enableMicrophone() {
        window.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
        // Let WebKit present its normal microphone prompt for this user action.
        // Stop the permission-check stream immediately; Record starts actual capture.
        web.callAsyncJavaScript("""
            const stream = await navigator.mediaDevices.getUserMedia({audio: true});
            stream.getTracks().forEach(track => track.stop());
            return true;
            """, arguments: [:], in: nil, in: .page) { result in
            switch result {
            case .success: self.web.reload()
            case .failure(let error):
                let alert = NSAlert()
                alert.messageText = "Microphone could not start"
                alert.informativeText = error.localizedDescription + " Allow TrueTone in System Settings > Privacy & Security > Microphone, or use Open in Browser or upload audio."
                alert.beginSheetModal(for: self.window)
            }
        }
    }
    @objc func grantProjectAccess() {
        guard let pathFile = Bundle.main.url(forResource: "project-path", withExtension: "txt"),
              let root = try? String(contentsOf: pathFile, encoding: .utf8).trimmingCharacters(in: .whitespacesAndNewlines) else { return }
        let panel = NSOpenPanel()
        panel.canChooseFiles = false
        panel.canChooseDirectories = true
        panel.allowsMultipleSelection = false
        panel.directoryURL = URL(fileURLWithPath: root)
        panel.message = "Select the TrueTone project folder to restore access after an app rebuild, then reopen TrueTone."
        panel.prompt = "Allow Project Access"
        panel.beginSheetModal(for: window) { response in
            if response == .OK, panel.url?.standardizedFileURL == URL(fileURLWithPath: root).standardizedFileURL {
                NSApp.terminate(nil)
            }
        }
    }
    func webView(_ webView: WKWebView, requestMediaCapturePermissionFor origin: WKSecurityOrigin,
                 initiatedByFrame frame: WKFrameInfo, type: WKMediaCaptureType,
                 decisionHandler: @escaping (WKPermissionDecision) -> Void) {
        // Only the owned localhost app may request audio. Never grant camera access.
        guard origin.protocol == "http", origin.host == "127.0.0.1", origin.port == 7861,
              frame.isMainFrame, type == .microphone else {
            decisionHandler(.deny)
            return
        }
        decisionHandler(.prompt)
    }
    func webViewWebContentProcessDidTerminate(_ webView: WKWebView) {
        if recoveredWebProcess {
            fail("The website renderer stopped. Use Reload Workspace or Open in Browser from the TrueTone menu.")
        } else {
            recoveredWebProcess = true
            reloadWorkspace()
        }
    }
    @objc func reloadWorkspace() {
        status.stringValue = "Reloading your local workspace…"
        loading.isHidden = false
        web.load(URLRequest(url: address, cachePolicy: .reloadIgnoringLocalCacheData))
    }
    @objc func closeWindow() { window.performClose(nil) }
    @objc func openBrowser() { NSWorkspace.shared.open(address) }
    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool { true }
    func applicationShouldTerminate(_ sender: NSApplication) -> NSApplication.TerminateReply {
        timer?.invalidate()
        finished = true
        guard let process = server, process.isRunning else { return .terminateNow }
        process.terminate()
        DispatchQueue.global().async {
            let deadline = Date().addingTimeInterval(3)
            while process.isRunning && Date() < deadline { Thread.sleep(forTimeInterval: 0.05) }
            if process.isRunning { kill(process.processIdentifier, SIGKILL) }
            process.waitUntilExit()
            DispatchQueue.main.async { sender.reply(toApplicationShouldTerminate: true) }
        }
        return .terminateLater
    }
}
let app = NSApplication.shared
let delegate = AppDelegate()
app.setActivationPolicy(.regular)
app.delegate = delegate
app.run()
