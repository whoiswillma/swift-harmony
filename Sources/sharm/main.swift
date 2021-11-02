import ArgumentParser
import OrderedCollections
import Foundation

enum LogLevel: String, ExpressibleByArgument {
    case trace
}

struct Sharm: ParsableCommand {

    static var configuration = CommandConfiguration(
        subcommands: [Interp.self, SMC.self, Compile.self],
        defaultSubcommand: Interp.self
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

    struct Interp: ParsableCommand {

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

    struct SMC: ParsableCommand {

        @OptionGroup
        var options: Options

        func run() throws {
            options.setLoggerLevel()

            let code = try options.readCodeFromHvmPath()
            let modelChecker = StatefulModelChecker(code: code)
            try modelChecker.run()
        }

    }

    struct Compile: ParsableCommand {

        @OptionGroup
        var options: Options

        func run() throws {
            options.setLoggerLevel()

            let code = try options.readCodeFromHvmPath()
            let compiler = H2SCompiler(code: code)
            try compiler.run()
        }

    }

}

Sharm.main()
