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

    static func generateSortedCollectionSwift(sharmSourcesDir: String) throws -> String {
        let sortedCollectionSwift = "\(sharmSourcesDir)/sharm/SortedCollections"

        let includes = try findSwiftFiles(in: sortedCollectionSwift)
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
            "\(sharmDir)/OpImpl.swift",
            "\(sharmDir)/State.swift",
        ]

        var contents = String()
        for include in includes {
            contents += try String(contentsOfFile: include)
        }
        return contents
    }

}

protocol H2SDefaultLineGenerator: OpFunction where Output == String {

}

extension H2SDefaultLineGenerator {

    func frame(name: String, params: VarTree, _ input: Input) -> String {
        "try OpImpl.frame(context: &context, name: \(String(reflecting: name)), params: \(String(reflecting: params)))"
    }

    func push(value: Value, _ input: Input) -> String {
        "try OpImpl.push(context: &context, value: \(String(reflecting: value)))"
    }

    func sequential(_ input: Input) -> String {
        "try OpImpl.sequential(context: &context)"
    }

    func choose(_ input: Input) -> String {
        "try OpImpl.choose(context: &context, chooseFn: { s in Int.random(in: 0..<s.count) })"
    }

    func storeVar(varTree: VarTree?, _ input: Input) -> String {
        "try OpImpl.storeVar(context: &context, varTree: \(String(reflecting: varTree)))"
    }

    func jump(pc: Int, _ input: Input) -> String {
        "try OpImpl.jump(context: &context, pc: \(String(reflecting: pc)))"
    }

    func jumpCond(pc: Int, cond: Value, _ input: Input) -> String {
        "try OpImpl.jumpCond(context: &context, pc: \(String(reflecting: pc)), cond: \(String(reflecting: cond)))"
    }

    func loadVar(varName: String?, _ input: Input) -> String {
        "try OpImpl.loadVar(context: &context, varName: \(String(reflecting: varName)))"
    }

    func address(_ input: Input) -> String {
        "try OpImpl.address(context: &context)"
    }

    func atomicInc(lazy: Bool, _ input: Input) -> String {
        "try OpImpl.atomicInc(context: &context, lazy: \(String(reflecting: lazy)))"
    }

    func atomicDec(_ input: Input) -> String {
        "try OpImpl.atomicDec(context: &context)"
    }

    func readonlyInc(_ input: Input) -> String {
        "try OpImpl.readonlyInc(context: &context)"
    }

    func readonlyDec(_ input: Input) -> String {
        "try OpImpl.readonlyDec(context: &context)"
    }

    func assertOp(_ input: Input) -> String {
        "try OpImpl.assertOp(context: &context)"
    }

    func delVar(varName: String?, _ input: Input) -> String {
        "try OpImpl.delVar(context: &context, varName: \(String(reflecting: varName)))"
    }

    func ret(_ input: Input) -> String {
        "try OpImpl.ret(context: &context)"
    }

    func apply(_ input: Input) -> String {
        "try OpImpl.apply(context: &context)"
    }

    func pop(_ input: Input) -> String {
        "try OpImpl.pop(context: &context)"
    }

    func cut(setName: String, key: VarTree?, value: VarTree, _ input: Input) -> String {
        "try OpImpl.cut(context: &context, setName: \(String(reflecting: setName)), key: \(String(reflecting: key)), value: \(String(reflecting: value)))"
    }

    func incVar(varName: String, _ input: Input) -> String {
        "try OpImpl.incVar(context: &context, varName: \(String(reflecting: varName)))"
    }

    func dup(_ input: Input) -> String {
        "try OpImpl.dup(context: &context)"
    }

    func split(count: Int, _ input: Input) -> String {
        "try OpImpl.split(context: &context, count: \(String(reflecting: count)))"
    }

    func move(offset: Int, _ input: Input) -> String {
        "try OpImpl.move(context: &context, offset: \(String(reflecting: offset)))"
    }

    func log(_ input: Input) -> String {
        "try OpImpl.log(context: &context)"
    }

}
