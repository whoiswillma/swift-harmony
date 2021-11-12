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
    }

    func run() throws {
        let orderedCollectionsSwift = try H2SUtil.generateOrderedCollectionSwift(sharmSourcesDir: sharmSourcesDir)
        try writer.write(orderedCollectionsSwift, filename: "OrderedCollections.swift")

        let sharmSwift = try H2SUtil.generateSharmSwift(sharmSourcesDir: sharmSourcesDir)
        try writer.write(sharmSwift, filename: "Sharm.swift")

        let mainSwift = generateMainSwift()
        try writer.write(mainSwift, filename: "main.swift")
    }

    private func generateMainSwift() -> String {
        generateBasicBlockStepFunction()
        + generateMain()
    }

    private func generateBasicBlockStepFunction() -> String {
        let blocks = getBasicBlockStartPCs().sorted().map(getBasicBlock(startPc:))

        var stepFn = """
        func basicBlockStep(
            context: inout Context,
            vars: inout HDict,
            contextBag: inout Bag<Context>
        ) throws {
            switch context.pc {

        """

        let genLine = H2SModelCheckerLineGenerator()
        for block in blocks {
            stepFn += "    case \(block.pc):\n"
            for op in block.ops {
                let lines = op.accept(genLine, ()).split(separator: "\n")
                for line in lines {
                    stepFn += "        \(line)\n"
                }
            }

            switch code[block.endPc] {
            case .atomicInc, .atomicDec, .load, .store:
                stepFn += "        throw Interrupt.switchPoint\n"

            case .frame, .push, .sequential, .choose, .storeVar, .jump, .jumpCond, .loadVar, .address, .nary,
                    .readonlyInc, .readonlyDec, .assertOp, .delVar, .ret, .spawn, .apply, .pop, .cut, .incVar, .dup,
                    .split, .move:
                break
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

    private func generateMain() -> String {
        return """
        enum Interrupt: Error {
            case choose(Int)
            case switchPoint
        }

        var visited: Set<State> = []
        var boundary: [State] = [.initialState]

        while var state = boundary.popLast() {
            if visited.contains(state) {
                continue
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

                case .switchPoint:
                    state.contextBag.remove(state.nextContextToRun)
                    state.contextBag.add(newContext)

                    for context in state.runnable {
                        var newState = state
                        newState.nextContextToRun = context
                        boundary.append(newState)
                    }
                }
            }
        }

        """
    }

    private func getBasicBlockStartPCs() -> Set<Int> {
        var startPCs: Set<Int> = [0]

        for (pc, op) in code.enumerated() {
            switch op {
            case .frame, .atomicInc, .atomicDec, .load, .store:
                startPCs.insert(pc)

            case .apply, .choose:
                startPCs.insert(pc + 1)

            case .jumpCond(pc: let newPc, cond: _):
                startPCs.insert(newPc)
                startPCs.insert(pc + 1)

            case .jump(pc: let newPc):
                startPCs.insert(newPc)

            case .push, .sequential, .storeVar, .loadVar, .address, .nary, .readonlyInc, .readonlyDec,
                    .assertOp, .delVar, .ret, .spawn, .pop, .cut, .incVar, .dup, .split, .move:

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
            pc += 1

            switch op {
            case .address, .assertOp, .cut, .delVar, .dup, .frame, .push, .sequential, .storeVar, .loadVar,
                    .nary, .readonlyInc, .readonlyDec, .spawn, .pop, .incVar, .store, .load, .atomicDec, .atomicInc,
                    .split, .move:

                basicBlock.append(op)

            case .jumpCond, .ret, .apply, .jump, .choose:
                basicBlock.append(op)
                break loop
            }

            switch nextOp {
            case .load, .store, .atomicInc, .atomicDec:
                break loop

            case .address, .assertOp, .choose, .cut, .delVar, .dup, .frame, .push, .sequential, .storeVar, .loadVar,
                    .nary, .readonlyInc, .readonlyDec, .spawn, .pop, .incVar, .jump, .jumpCond, .ret, .apply, .split,
                    .move, nil:

                break
            }
        }

        return BasicBlock(pc: startPc, ops: basicBlock, endPc: pc)
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

}
