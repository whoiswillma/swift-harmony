import ArgumentParser
import Foundation

enum LogLevel: String, ExpressibleByArgument {
    case trace
}

struct Sharm: ParsableCommand {

    static var configuration = CommandConfiguration(
        subcommands: [IE.self, IMC.self, CE.self, CMC.self]
    )

}

struct Options: ParsableCommand {

    @Option
    var logLevel: LogLevel?

    @Argument
    var hvmPath: String = ""

    func setLoggerLevel() {
        switch logLevel {
        case .trace: logger.logLevel = .trace
        case nil: break
        }
    }

    func readCodeFromHvmPath() throws -> [Op] {
        struct HVMFile: Decodable {
            let code: [Op]
        }

        let url = URL(fileURLWithPath: hvmPath)
        let hvmData = try Data(contentsOf: url)
        let hvmFile = try JSONDecoder().decode(HVMFile.self, from: hvmData)
        return hvmFile.code
    }

}

extension Sharm {

    struct IE: ParsableCommand {

        @Flag
        var printHistory: Bool = false

        @OptionGroup
        var options: Options

        func run() throws {
            options.setLoggerLevel()

            let code = try options.readCodeFromHvmPath()
            let nondeterminism = BookkeepingNondeterminism()
            let interpreter = Interpreter(code: code, nondeterminism: nondeterminism)

            do {
                try interpreter.run()
            } catch {
                print("History")
                for elem in nondeterminism.history {
                    switch elem {
                    case .index(let i, let s): print("\tChose \(i) out of \(s)")
                    case .context(let i, let s): print("\tChose \(i) out of \(s)")
                    }
                }

                throw error
            }
        }

    }

    struct IMC: ParsableCommand {

        @OptionGroup
        var options: Options

        func run() throws {
            options.setLoggerLevel()

            let code = try options.readCodeFromHvmPath()
            let modelChecker = StatefulModelChecker(code: code)
            try modelChecker.run()
        }

    }

    struct CE: ParsableCommand {

        @OptionGroup
        var options: Options

        @Option
        var sharmSourcesDir = "\(FileManager.default.currentDirectoryPath)/Sources"

        @Option
        var outputDir = "\(FileManager.default.currentDirectoryPath)/sharmcompile"

        @Flag
        var dryRun = false

        @Flag
        var genSanityChecks = false

        func run() throws {
            options.setLoggerLevel()

            let code = try options.readCodeFromHvmPath()
            let compiler = H2SCompiler(
                code: code,
                sharmSourcesDir: sharmSourcesDir,
                outputDir: outputDir,
                dryRun: dryRun,
                genSanityChecks: genSanityChecks
            )
            try compiler.run()
        }

    }

    struct CMC: ParsableCommand {

        @OptionGroup
        var options: Options

        @Option
        var sharmSourcesDir = "\(FileManager.default.currentDirectoryPath)/Sources"

        @Option
        var outputDir = "\(FileManager.default.currentDirectoryPath)/sharmcompile"

        @Flag
        var dryRun = false

        @Flag
        var genSanityChecks = false

        func run() throws {
            options.setLoggerLevel()

            let code = try options.readCodeFromHvmPath()
            let compiler = H2SModelChecker(
                code: code,
                sharmSourcesDir: sharmSourcesDir,
                outputDir: outputDir,
                dryRun: dryRun,
                genSanityChecks: genSanityChecks
            )
            try compiler.run()
        }

    }

}

Sharm.main()
