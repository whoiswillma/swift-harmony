#!/usr/bin/env swift

import Foundation
import OrderedCollections

private class BasicNondeterminism: Nondeterminism {

    enum History {
        case index(Int, Set)
        case context(String, [String])
    }

    var history: [History] = []

    func chooseIndex(_ values: Set) -> Int {
        let index = Int.random(in: 0..<values.count)
        history.append(.index(index, values))
        return index
    }

    func chooseContext(_ contexts: [Context]) -> Int {
        let index = Int.random(in: 0..<contexts.count)
        let name = contexts[index].name
        history.append(.context(name, contexts.map(\.name)))
        return index
    }

}

private enum InterpreterInterrupt: Error {
    case switchPoint
    case spawn(Context)
    case updateContextBag
}

private struct InterpreterOpVisitor: DeterministicContextOpVisitor {

    var context: Context
    unowned let interpreter: Interpreter

    init(context: Context, interpreter: Interpreter) {
        self.context = context
        self.interpreter = interpreter
    }

    mutating func choose() throws {
        try OpImpl.choose(context: &context, nondeterminism: interpreter.nondeterminism)
    }

    mutating func store(_ addressOrNil: Value?) throws {
        try OpImpl.store(context: &context, vars: &interpreter.vars, addressOrNil)
        throw InterpreterInterrupt.switchPoint
    }

    mutating func load(_ addressOrNil: Value?) throws {
        try OpImpl.load(context: &context, vars: &interpreter.vars, addressOrNil)
        throw InterpreterInterrupt.switchPoint
    }

    mutating func spawn(eternal: Bool) throws {
        let child = try OpImpl.spawn(parent: &context, name: "T\(interpreter.spawnCounter)", eternal: eternal)
        interpreter.spawnCounter += 1
        throw InterpreterInterrupt.spawn(child)
    }

    mutating func nary(_ nary: Nary) throws {
        try OpImpl.nary(context: &context, contextBag: interpreter.contextBag, nary)
    }

}

private class Interpreter {

    let code: [Op]

    var spawnCounter: Int = 0
    var nondeterminism: BasicNondeterminism
    var vars: Dict
    var contextBag: Bag<Context>

    var nonterminatedContexts: Swift.Set<Context> {
        contextBag.elements().filter { !$0.terminated }
    }

    var allTerminated: Bool {
        nonterminatedContexts.isEmpty
    }

    init(code: [Op]) {
        self.code = code
        let initContext = Context(name: "__init__", entry: 0, arg: .dict([:]), stack: [.dict([:])])
        self.nondeterminism = BasicNondeterminism()
        self.vars = Dict()
        self.contextBag = Bag([initContext])
    }

    private func getRunnable() -> [Context] {
        let contexts = nonterminatedContexts
        if let context = contexts.first(where: { $0.isAtomic }) {
            return [context]
        } else {
            return contexts.sorted()
        }
    }

    func step() throws {
        let runnable = getRunnable()
        let index: Int
        if runnable.count == 1 {
            index = 0
        } else {
            index = nondeterminism.chooseContext(runnable)
        }
        var context = runnable[index]

        var visitor = InterpreterOpVisitor(context: context, interpreter: self)
        do {
            while !visitor.context.terminated {
                do {
                    print(visitor.context.name, code[visitor.context.pc], visitor.context.stack)
                    if !visitor.context.isAtomic, case .atomicInc = code[visitor.context.pc] {
                        contextBag.remove(context)
                        contextBag.add(visitor.context)
                        context = visitor.context
                    }
                    try code[visitor.context.pc].accept(&visitor)
                } catch InterpreterInterrupt.spawn(let context) {
                    self.contextBag.add(context)
                }
            }
        } catch InterpreterInterrupt.switchPoint {

        } catch {
            throw error
        }

        contextBag.remove(context)
        contextBag.add(visitor.context)
    }

}

struct HVM: Decodable {
    let code: [Op]
}

let url = URL(fileURLWithPath: "/Users/williamma/Documents/sharm/Diners.hvm")
let hvmData = try Data(contentsOf: url)
let hvm = try JSONDecoder().decode(HVM.self, from: hvmData)

private let interpreter = Interpreter(code: hvm.code)
do {
    while !interpreter.allTerminated {
        try interpreter.step()
    }
} catch {
    print("History")
    for elem in interpreter.nondeterminism.history {
        switch elem {
        case .index(let i, let s): print("\tChose \(i) out of \(s)")
        case .context(let i, let s): print("\tChose \(i) out of \(s)")
        }
    }

    throw error
}
