//
//  Interpreter.swift
//  
//
//  Created by William Ma on 10/31/21.
//

import Foundation

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

    mutating func atomicInc(lazy: Bool) throws {
        try OpImpl.atomicInc(context: &context, lazy: lazy)
    }

    mutating func atomicDec() throws {
        try OpImpl.atomicDec(context: &context)

        if !context.isAtomic {
            throw InterpreterInterrupt.switchPoint
        }
    }

    mutating func choose() throws {
        try OpImpl.choose(context: &context, nondeterminism: interpreter.nondeterminism)
    }

    mutating func store(address: Value?) throws {
        try OpImpl.store(context: &context, vars: &interpreter.vars, address: address)
        throw InterpreterInterrupt.switchPoint
    }

    mutating func load(address: Value?) throws {
        try OpImpl.load(context: &context, vars: &interpreter.vars, address: address)
        throw InterpreterInterrupt.switchPoint
    }

    mutating func spawn(eternal: Bool) throws {
        let child = try OpImpl.spawn(
            parent: &context,
            name: "T\(interpreter.spawnCounter)",
            eternal: eternal
        )
        interpreter.spawnCounter += 1
        throw InterpreterInterrupt.spawn(child)
    }

    mutating func nary(_ nary: Nary) throws {
        try OpImpl.nary(context: &context, contextBag: interpreter.contextBag, nary: nary)
    }

}

class Interpreter {

    let code: [Op]

    var spawnCounter: Int = 0
    var nondeterminism: Nondeterminism
    var vars: Dict
    var contextBag: Bag<Context>

    var nonterminatedContexts: Swift.Set<Context> {
        contextBag.elements().filter { !$0.terminated }
    }

    var allTerminated: Bool {
        nonterminatedContexts.isEmpty
    }

    init(code: [Op], nondeterminism: Nondeterminism) {
        self.code = code
        let initContext = Context(name: "__init__", entry: 0, arg: .dict([:]), stack: [.dict([:])])
        self.nondeterminism = nondeterminism
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
        let context = runnable[index]

        var visitor = InterpreterOpVisitor(context: context, interpreter: self)
        do {
            var firstInstruction = true
            while !visitor.context.terminated {
                do {
                    logger.trace("\(visitor.context.name), \(code[visitor.context.pc]), \(visitor.context.stack)")

                    if firstInstruction, !visitor.context.isAtomic, case .atomicInc = code[visitor.context.pc] {
                        throw InterpreterInterrupt.switchPoint
                    }
                    try code[visitor.context.pc].accept(&visitor)
                    firstInstruction = false
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
