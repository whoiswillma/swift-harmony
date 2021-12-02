//
//  H2SCompiler.swift
//  
//
//  Created by William Ma on 11/2/21.
//

import Foundation

class H2SCompiler {

    private let code: [Op]
    private let sharmSourcesDir: String
    private let writer: H2SUtil.FileWriter
    private let genSanityChecks: Bool
    private let outputDir: String

    init(
        code: [Op],
        sharmSourcesDir: String,
        outputDir: String,
        dryRun: Bool,
        genSanityChecks: Bool
    ) {
        self.code = code
        self.sharmSourcesDir = sharmSourcesDir
        self.writer = H2SUtil.FileWriter(outputDir: outputDir, dryRun: dryRun)
        self.genSanityChecks = genSanityChecks
        self.outputDir = outputDir
    }

    func run() throws {
        try FileManager.default.createDirectory(atPath: outputDir, withIntermediateDirectories: true, attributes: nil)

        let orderedCollectionsSwift = try H2SUtil.generateOrderedCollectionSwift(sharmSourcesDir: sharmSourcesDir)
        try writer.write(orderedCollectionsSwift, filename: "OrderedCollections.swift")

        let sharmSwift = try H2SUtil.generateSharmSwift(sharmSourcesDir: sharmSourcesDir)
        try writer.write(sharmSwift, filename: "Sharm.swift")

        let mainSwift = generateMainSwift()
        try writer.write(mainSwift, filename: "main.swift")

        try writer.write("""
        swiftc *.swift -o main
        ./main

        """, filename: "run.sh")
        try FileManager.default.setAttributes([.posixPermissions:0o777], ofItemAtPath: "\(outputDir)/run.sh")
    }

    func generateStepFunction() -> String {
        let blocks = getBasicBlockStartPCs().sorted().map(getBasicBlock(startPc:))

        var stepFn = """

        func opLog(context: inout Context) throws {
            guard let value = context.stack.popLast() else {
                throw OpError.stackIsEmpty
            }

            print(value.description)
            context.pc += 1
        }

        func step(
            context: inout Context,
            vars: inout HDict,
            contextBag: inout Bag<Context>,
            threadCount: inout Int
        ) throws {
            switch context.pc {

        """

        let genLine = H2SCompilerLineGenerator()
        for block in blocks {
            stepFn += "    case \(block.pc):\n"
            for (i, op) in block.ops.enumerated() {
                if genSanityChecks {
                    stepFn += ##"""
                            print(
                                context.name,
                                #"\##(op)"#,
                                "[\(context.stack.map { $0.description }.joined(separator: ", "))]"
                            )\##n
                    """##
                }
                if genSanityChecks {
                    stepFn += "        print(context.name, vars)\n"
                }
                if genSanityChecks {
                    stepFn += "        print()\n"
                }
                if genSanityChecks {
                    stepFn += "        assert(context.pc == \(block.pc + i))\n"
                }
                let lines = op.accept(genLine, ()).split(separator: "\n")
                for line in lines {
                    stepFn += "        \(line)\n"
                }
            }
        }

        stepFn += #"""
            default:
                fatalError("pc: \(context.pc)")
            }
        }


        """#

        return stepFn
    }

    private func generateMainSwift() -> String {
        generateStepFunction() + generateMain()
    }

    private func generateMain() -> String {
        return """

        var vars = HDict()
        var contextBag = Bag<Context>([.initContext])
        var threadCount: Int = 0

        func nonterminatedContexts() -> Set<Context> {
            contextBag.elements().filter { !$0.terminated }
        }

        func getRunnable() -> [Context] {
            let contexts = nonterminatedContexts()

            let atomicContexts = contexts.filter { $0.isAtomic }
            assert(atomicContexts.count <= 1)

            if let context = atomicContexts.first {
                return [context]
            } else {
                return contexts.sorted(by: { $0.name < $1.name })
            }
        }

        while let context = getRunnable().randomElement() {
            var newContext = context
            try step(context: &newContext, vars: &vars, contextBag: &contextBag, threadCount: &threadCount)
            contextBag.remove(context)
            contextBag.add(newContext)
        }

        """
    }

    private func getBasicBlockStartPCs() -> Set<Int> {
        var startPCs: Set<Int> = [0]

        for (pc, op) in code.enumerated() {
            switch op {
            case .frame, .atomicInc, .atomicDec, .load, .store:
                startPCs.insert(pc)

            case .apply:
                startPCs.insert(pc + 1)

            case .jumpCond(pc: let newPc, cond: _):
                startPCs.insert(newPc)
                startPCs.insert(pc + 1)

            case .jump(pc: let newPc):
                startPCs.insert(newPc)

            case .push, .sequential, .choose, .storeVar, .loadVar, .address, .nary, .readonlyInc, .readonlyDec,
                    .assertOp, .delVar, .ret, .spawn, .pop, .cut, .incVar, .dup, .split, .move, .log:
                break
            }
        }

        return startPCs
    }

    private func getBasicBlock(startPc: Int) -> BasicBlock {
        var basicBlock = [Op]()
        var pc = startPc

        loop: while pc < code.count {
            let op = code[pc]
            let nextOp: Op?
            if pc < code.count {
                nextOp = code[pc + 1]
            } else {
                nextOp = nil
            }
            basicBlock.append(op)
            pc += 1

            switch op {
            case .jumpCond, .ret, .apply, .jump:
                break loop

            case .address, .assertOp, .choose, .cut, .delVar, .dup, .frame, .push, .sequential, .storeVar, .loadVar,
                    .nary, .readonlyInc, .readonlyDec, .spawn, .pop, .incVar, .store, .load, .atomicDec, .atomicInc,
                    .split, .move, .log:
                break
            }

            switch nextOp {
            case .load, .store, .atomicInc, .atomicDec:
                break loop

            case .address, .assertOp, .choose, .cut, .delVar, .dup, .frame, .push, .sequential, .storeVar, .loadVar,
                    .nary, .readonlyInc, .readonlyDec, .spawn, .pop, .incVar, .jump, .jumpCond, .ret, .apply, .split,
                    .move, .log, nil:
                break
            }
        }

        return BasicBlock(pc: startPc, ops: basicBlock)
    }

}

private struct H2SCompilerLineGenerator: H2SDefaultLineGenerator {

    func spawn(eternal: Bool, _ input: Void) -> String {
        #"""
        try contextBag.add(OpImpl.spawn(parent: &context, name: "T\(threadCount)", eternal: \#(String(reflecting: eternal))))
        threadCount += 1
        """#
    }

    func nary(nary: Nary, _ input: Void) -> String {
        "try OpImpl.nary(context: &context, contextBag: contextBag, nary: \(String(reflecting: nary)))"
    }

    func load(address: Value?, _ input: Void) -> String {
        "try OpImpl.load(context: &context, vars: &vars, address: \(String(reflecting: address)))"
    }

    func store(address: Value?, _ input: Void) -> String {
        "try OpImpl.store(context: &context, vars: &vars, address: \(String(reflecting: address)))"
    }

    func log(_ input: Void) -> String {
        "try opLog(context: &context)"
    }

    func atomicInc(lazy: Bool, _ input: Void) -> String {
        "try OpImpl.atomicInc(context: &context, lazy: false)"
    }

}
