//
//  H2SUtil.swift
//  
//
//  Created by William Ma on 11/9/21.
//

import Foundation

enum H2SUtil {

    static func findSwiftFiles(in directory: String) throws -> [String] {
        let contents = try FileManager.default.contentsOfDirectory(atPath: directory)
        var results: [String] = []
        for content in contents {
            let path = "\(directory)/\(content)"
            var isDirectory: ObjCBool = false
            if FileManager.default.fileExists(atPath: path, isDirectory: &isDirectory) && isDirectory.boolValue {
                results.append(contentsOf: try findSwiftFiles(in: path))
            } else if path.hasSuffix(".swift") {
                results.append(path)
            }
        }
        return results
    }

    struct FileWriter {

        let outputDir: String
        let dryRun: Bool

        func write(_ str: String, filename: String) throws {
            let path = "\(outputDir)/\(filename)"
            if dryRun {
                logger.info("Dry Run: Would have wrote to \(path)")
            } else {
                try str.write(toFile: "\(outputDir)/\(filename)", atomically: true, encoding: .utf8)
            }
        }

    }

}

extension H2SUtil {

    static func generateOrderedCollectionSwift(sharmSourcesDir: String) throws -> String {
        let orderedCollectionsDir = "\(sharmSourcesDir)/sharm/OrderedCollections"

        let includes = try findSwiftFiles(in: orderedCollectionsDir)
        var contents = String()
        for include in includes {
            contents += try String(contentsOfFile: include)
        }
        return contents
    }

}

extension H2SUtil {

    static func generateSharmSwift(sharmSourcesDir: String) throws -> String {
        let sharmDir = "\(sharmSourcesDir)/sharm"
        let includes = [
            "\(sharmDir)/Bag.swift",
            "\(sharmDir)/VarTree.swift",
            "\(sharmDir)/Nary.swift",
            "\(sharmDir)/Value.swift",
            "\(sharmDir)/Op.swift",
        ]

        var contents = String()
        for include in includes {
            contents += try String(contentsOfFile: include)
        }
        return contents
    }

}

extension H2SUtil {

    private struct BasicBlock {
        let pc: Int
        let ops: [Op]
    }

    private static func getBasicBlocks(ops: [Op]) -> [BasicBlock] {
        let startPCs = getBasicBlockStartPCs(ops: ops)

        var basicBlocks = [BasicBlock]()
        for pc in startPCs.sorted() {
            let ops = getBasicBlockOps(ops: ops, startPc: pc)
            let basicBlock = BasicBlock(pc: pc, ops: ops)
            basicBlocks.append(basicBlock)
        }

        return basicBlocks
    }

    private static func getBasicBlockStartPCs(ops: [Op]) -> Set<Int> {
        var startPCs: Set<Int> = [0]

        for (pc, op) in ops.enumerated() {
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

            case .push(value: .pc(let newPc)):
                startPCs.insert(newPc)

            case .push,
                 .sequential,
                 .choose,
                 .storeVar,
                 .loadVar,
                 .address,
                 .nary,
                 .readonlyInc,
                 .readonlyDec,
                 .assertOp,
                 .delVar,
                 .ret,
                 .spawn,
                 .pop,
                 .cut,
                 .incVar,
                 .dup,
                 .split,
                 .move:

                break
            }
        }

        return startPCs
    }

    private static func getBasicBlockOps(ops: [Op], startPc: Int) -> [Op] {
        var basicBlock = [Op]()
        var pc = startPc

        loop: while pc < ops.count {
            let op = ops[pc]
            let nextOp: Op?
            if pc < ops.count {
                nextOp = ops[pc + 1]
            } else {
                nextOp = nil
            }

            switch op {
            case .address,
                 .assertOp,
                 .choose,
                 .cut,
                 .delVar,
                 .dup,
                 .frame,
                 .push,
                 .sequential,
                 .storeVar,
                 .loadVar,
                 .nary,
                 .readonlyInc,
                 .readonlyDec,
                 .spawn,
                 .pop,
                 .incVar,
                 .store,
                 .load,
                 .atomicDec,
                 .atomicInc,
                 .split,
                 .move:

                basicBlock.append(op)
                pc += 1

            case .jumpCond, .ret, .apply, .jump:
                basicBlock.append(op)
                break loop
            }

            switch nextOp {
            case .load,
                 .store,
                 .atomicInc,
                 .atomicDec:
                break loop

            case .address,
                 .assertOp,
                 .choose,
                 .cut,
                 .delVar,
                 .dup,
                 .frame,
                 .push,
                 .sequential,
                 .storeVar,
                 .loadVar,
                 .nary,
                 .readonlyInc,
                 .readonlyDec,
                 .spawn,
                 .pop,
                 .incVar,
                 .jump,
                 .jumpCond,
                 .ret,
                 .apply,
                 .split,
                 .move,
                 nil:

                break
            }
        }

        return basicBlock
    }

    private static func generateLine(op: Op) -> String {
        switch op {
        case .frame(name: let name, params: let params):
            return "try OpImpl.frame(context: &context, name: \(String(reflecting: name)), params: \(String(reflecting: params)))"

        case .push(value: let value):
            return "try OpImpl.push(context: &context, value: \(String(reflecting: value)))"

        case .sequential:
            return "try OpImpl.sequential(context: &context)"

        case .choose:
            return "try OpImpl.choose(context: &context, chooseFn: { s in Int.random(in: 0..<s.count) })"

        case .store(address: let address):
            return "try OpImpl.store(context: &context, vars: &vars, address: \(String(reflecting: address)))"

        case .storeVar(varTree: let varTree):
            return "try OpImpl.storeVar(context: &context, varTree: \(String(reflecting: varTree)))"

        case .jump(pc: let pc):
            return "try OpImpl.jump(context: &context, pc: \(String(reflecting: pc)))"

        case .jumpCond(pc: let pc, cond: let cond):
            return "try OpImpl.jumpCond(context: &context, pc: \(String(reflecting: pc)), cond: \(String(reflecting: cond)))"

        case .loadVar(varName: let varName):
            return "try OpImpl.loadVar(context: &context, varName: \(String(reflecting: varName)))"

        case .load(address: let address):
            return "try OpImpl.load(context: &context, vars: &vars, address: \(String(reflecting: address)))"

        case .address:
            return "try OpImpl.address(context: &context)"

        case .nary(nary: let nary):
            return "try OpImpl.nary(context: &context, contextBag: contextBag, nary: \(String(reflecting: nary)))"

        case .atomicInc(lazy: let lazy):
            return "try OpImpl.atomicInc(context: &context, lazy: \(String(reflecting: lazy)))"

        case .atomicDec:
            return "try OpImpl.atomicDec(context: &context)"

        case .readonlyInc:
            return "try OpImpl.readonlyInc(context: &context)"

        case .readonlyDec:
            return "try OpImpl.readonlyDec(context: &context)"

        case .assertOp:
            return "try OpImpl.assertOp(context: &context)"

        case .delVar(varName: let varName):
            return "try OpImpl.delVar(context: &context, varName: \(String(reflecting: varName)))"

        case .ret:
            return "try OpImpl.ret(context: &context)"

        case .spawn(eternal: let eternal):
            return #"""
            try contextBag.add(OpImpl.spawn(parent: &context, name: "T\(threadCount)", eternal: \#(String(reflecting: eternal))))
            threadCount += 1
            """#

        case .apply:
            return "try OpImpl.apply(context: &context)"

        case .pop:
            return "try OpImpl.pop(context: &context)"

        case .cut(setName: let setName, varTree: let varTree):
            return "try OpImpl.cut(context: &context, setName: \(String(reflecting: setName)), varTree: \(String(reflecting: varTree)))"

        case .incVar(varName: let varName):
            return "try OpImpl.incVar(context: &context, varName: \(String(reflecting: varName)))"

        case .dup:
            return "try OpImpl.dup(context: &context)"

        case .split(count: let count):
            return "try OpImpl.split(context: &context, count: \(String(reflecting: count)))"

        case .move(offset: let offset):
            return "try OpImpl.move(context: &context, offset: \(String(reflecting: offset)))"
        }
    }

    static func generateStepFunction(ops: [Op], enableAssertions: Bool, printContext: Bool, printVars: Bool) -> String {
        let blocks = getBasicBlocks(ops: ops)
        var stepFn = """
        func step(context: inout Context) throws {
            switch context.pc {

        """

        for block in blocks {
            stepFn += "    case \(block.pc):\n"
            for (i, op) in block.ops.enumerated() {
                if printContext {
                    stepFn += ##"""
                            print(
                                context.name,
                                #"\##(op)"#,
                                "[\(context.stack.map { $0.description }.joined(separator: ", "))]"
                            )\##n
                    """##
                }
                if printVars {
                    stepFn += "        print(context.name, vars)\n"
                }
                if printContext || printVars {
                    stepFn += "        print()\n"
                }
                if enableAssertions {
                    stepFn += "        assert(context.pc == \(block.pc + i))\n"
                }
                let lines = generateLine(op: op).split(separator: "\n")
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

}
