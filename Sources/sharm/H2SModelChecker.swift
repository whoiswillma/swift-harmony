//
//  H2SModelChecker.swift
//  
//
//  Created by William Ma on 11/9/21.
//

import Foundation

class H2SModelChecker {

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

    private func generateMainSwift() -> String {
        generateCodeArray()
        + generateBasicBlockStepFunction()
        + generateMain()
    }

    private func generateCodeArray() -> String {
        """
        let code = [
            \(code.map { $0.debugDescription }.joined(separator: ",\n    "))
        ]


        """
    }

    private func generateBasicBlockStepFunction() -> String {
        let blocks = getBasicBlockStartPCs().sorted().map(getBasicBlock(startPc:))

        var stepFn = """

        func checkSwitchPoint(context: inout Context) throws {
            switch code[context.pc] {
                case .atomicInc(lazy: false):
                    throw Interrupt.switchPoint

                case .load, .store:
                    if !context.isAtomic {
                        if context.atomicLevel > 0 {
                            context.isAtomic = true
                        }
                        throw Interrupt.switchPoint
                    }

                case .frame, .push, .sequential, .choose, .storeVar, .jump, .jumpCond, .loadVar, .address, .nary,
                        .readonlyInc, .readonlyDec, .assertOp, .delVar, .ret, .spawn, .apply, .pop, .cut, .incVar, .dup,
                        .split, .move, .atomicInc(lazy: true), .atomicDec:
                    break
            }
        }

        func basicBlockStep(
            context: inout Context,
            vars: inout HDict,
            contextBag: inout Bag<Context>
        ) throws {
            switch context.pc {

        """

        let genLine = H2SModelCheckerLineGenerator()
        for block in blocks {
            stepFn += """
                    case \(block.pc):

            """
            for op in block.ops {
                let lines = op.accept(genLine, ()).split(separator: "\n")
                for line in lines {
                    stepFn += ##"""
                                #if DEBUG
                                print(context.pc, context.stack, #"\##(String(reflecting: op))"#)
                                #endif

                    """##
                    stepFn += """
                                \(line)

                    """
                }
            }
            stepFn += """
                        try checkSwitchPoint(context: &context)

            """
        }

        stepFn += #"""
            default:
                fatalError("pc: \(context.pc)")
            }
        }


        """#

        return stepFn
    }

    private func generateMain() -> String {
        return #"""

        OpImpl.printEnabled = false

        enum Interrupt: Error {
            case choose(Int)
            case switchPoint
        }

        var stutterSteps: Int = 0
        var visited: Set<State> = []
        var boundary: [State] = [.initialState]

        while var state = boundary.popLast() {
            if visited.contains(state) {
                continue
            }

            if visited.count % 1000 == 0 {
                print(visited.count, boundary.count)
            }

            visited.insert(state)
            var newContext = state.nextContextToRun
            do {
                while !newContext.terminated {
                    try basicBlockStep(
                        context: &newContext,
                        vars: &state.vars,
                        contextBag: &state.contextBag
                    )
                }

                throw Interrupt.switchPoint

            } catch let i as Interrupt {
                #if DEBUG
                print("Switch point: \(newContext.pc)")
                #endif

                switch i {
                case .choose(let count):
                    state.contextBag.remove(state.nextContextToRun)

                    for i in 0..<count {
                        var context = newContext
                        try OpImpl.choose(context: &context, chooseFn: { _ in i })
                        var newState = state
                        newState.contextBag.add(context)
                        newState.nextContextToRun = context
                        boundary.append(newState)
                    }

                    if count == 1 {
                        stutterSteps += 1
                    }

                case .switchPoint:
                    state.contextBag.remove(state.nextContextToRun)
                    state.contextBag.add(newContext)

                    for context in state.runnable {
                        var newState = state
                        newState.nextContextToRun = context
                        boundary.append(newState)
                    }

                    if state.runnable.count == 1 {
                        stutterSteps += 1
                    }
                }

            } catch let e as OpError {
                fatalError("Error while model checking: \(e)")

            } catch {
                dump(state)
                fatalError("\(error)")
            }
        }

        print("No errors found")
        print("Total states: \(visited.count)")
        print("Stutter steps: \(stutterSteps)")
        """#
    }

    private func getBasicBlockStartPCs() -> Set<Int> {
        var startPCs: Set<Int> = [0]

        for (pc, op) in code.enumerated() {
            switch op {
            case .frame, .atomicDec, .load, .store:
                startPCs.insert(pc)

            case .atomicInc(lazy: false):
                startPCs.insert(pc)
                startPCs.insert(pc + 1)

            case .apply, .choose:
                startPCs.insert(pc + 1)

            case .jumpCond(pc: let newPc, cond: _):
                startPCs.insert(newPc)
                startPCs.insert(pc + 1)

            case .jump(pc: let newPc):
                startPCs.insert(newPc)

            case .push, .sequential, .storeVar, .loadVar, .address, .nary, .readonlyInc, .readonlyDec,
                    .assertOp, .delVar, .ret, .spawn, .pop, .cut, .incVar, .dup, .split, .move, .atomicInc(lazy: true),
                    .log:

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
            case .jumpCond, .ret, .apply, .jump, .choose, .atomicInc(lazy: false):
                break loop

            case .address, .assertOp, .cut, .delVar, .dup, .frame, .push, .sequential, .storeVar, .loadVar,
                    .nary, .readonlyInc, .readonlyDec, .spawn, .pop, .incVar, .store, .load, .atomicDec,
                    .atomicInc(lazy: true), .split, .move, .log:
                break
            }

            switch nextOp {
            case .load, .store, .atomicInc(lazy: false), .atomicDec:
                break loop

            case .address, .assertOp, .choose, .cut, .delVar, .dup, .frame, .push, .sequential, .storeVar, .loadVar,
                    .nary, .readonlyInc, .readonlyDec, .spawn, .pop, .incVar, .jump, .jumpCond, .ret, .apply, .split,
                    .move, .atomicInc(lazy: true), .log, nil:
                break
            }
        }

        return BasicBlock(pc: startPc, ops: basicBlock)
    }

}

private struct H2SModelCheckerLineGenerator: H2SDefaultLineGenerator {

    func spawn(eternal: Bool, _ input: Void) -> String {
        #"""
        try contextBag.add(OpImpl.spawn(parent: &context, name: "", eternal: \#(String(reflecting: eternal))))
        """#
    }

    func nary(nary: Nary, _ input: Void) -> String {
        "try OpImpl.nary(context: &context, contextBag: contextBag, nary: \(String(reflecting: nary)))"
    }

    func load(address: Value?, _ input: Void) -> String {
        """
        try OpImpl.load(context: &context, vars: &vars, address: \(String(reflecting: address)))
        """
    }

    func store(address: Value?, _ input: Void) -> String {
        """
        try OpImpl.store(context: &context, vars: &vars, address: \(String(reflecting: address)))
        """
    }

    func choose(_ input: Void) -> String {
        """
        guard case let .set(s) = context.stack.last else { throw OpError.stackTypeMismatch(expected: .set) }
        throw Interrupt.choose(s.count)
        """
    }

    func atomicInc(lazy: Bool, _ input: Void) -> String {
        if lazy {
            return """
            try OpImpl.atomicInc(context: &context, lazy: true)
            """
        } else {
            return """
            try OpImpl.atomicInc(context: &context, lazy: false)
            throw Interrupt.switchPoint
            """
        }
    }

}
