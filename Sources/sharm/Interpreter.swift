//
//  Interpreter.swift
//  
//
//  Created by William Ma on 10/31/21.
//

private enum InterpreterInterrupt: Error {
    case switchPoint
    case spawn(Context)
}

private struct InterpreterOpVisitor: DefaultOpImplVisitor {

    typealias Input = Void
    typealias Output = Void

    var context: Context
    unowned let interpreter: Interpreter

    init(context: Context, interpreter: Interpreter) {
        self.context = context
        self.interpreter = interpreter
    }

    mutating func atomicInc(lazy: Bool, _ input: Void) throws {
        try OpImpl.atomicInc(context: &context, lazy: false)

        if context.isAtomic {
            throw InterpreterInterrupt.switchPoint
        }
    }

    mutating func atomicDec(_ input: Void) throws {
        try OpImpl.atomicDec(context: &context)

        if !context.isAtomic {
            throw InterpreterInterrupt.switchPoint
        }
    }

    mutating func choose(_ input: Void) throws {
        try OpImpl.choose(context: &context, chooseFn: interpreter.nondeterminism.chooseIndex)
    }

    mutating func store(address: Value?, _ input: Void) throws {
        try OpImpl.store(context: &context, vars: &interpreter.vars, address: address)
        throw InterpreterInterrupt.switchPoint
    }

    mutating func load(address: Value?, _ input: Void) throws {
        try OpImpl.load(context: &context, vars: &interpreter.vars, address: address)
        throw InterpreterInterrupt.switchPoint
    }

    mutating func spawn(eternal: Bool, _ input: Void) throws {
        let child = try OpImpl.spawn(
            parent: &context,
            name: "T\(interpreter.spawnCounter)",
            eternal: eternal
        )
        interpreter.spawnCounter += 1
        throw InterpreterInterrupt.spawn(child)
    }

    mutating func nary(nary: Nary, _ input: Void) throws {
        try OpImpl.nary(context: &context, contextBag: interpreter.contextBag, nary: nary)
    }

    mutating func log(_ input: Void) throws -> Void {
        guard let value = context.stack.popLast() else {
            throw OpError.stackIsEmpty
        }

        print(value.description)
        context.pc += 1
    }

}

class Interpreter {

    let code: [Op]

    var spawnCounter: Int = 0
    var nondeterminism: Nondeterminism
    var vars: HDict
    var contextBag: Bag<Context>

    var nonterminatedContexts: Set<Context> {
        contextBag.elements().filter { !$0.terminated }
    }

    var allTerminated: Bool {
        nonterminatedContexts.isEmpty
    }

    init(code: [Op], nondeterminism: Nondeterminism) {
        self.code = code
        self.nondeterminism = nondeterminism
        self.vars = HDict()
        self.contextBag = Bag([.initContext])
    }

    private func getRunnable() -> [Context] {
        let contexts = nonterminatedContexts

        if let context = contexts.first(where: { $0.isAtomic }) {
            return [context]
        } else {
            return contexts.sorted(by: { $0.name < $1.name })
        }
    }

    func run() throws {
        while !allTerminated {
            try step()
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
            repeat {
                do {
                    logger.trace("\(code[visitor.context.pc]), \(visitor.context)")
                    try code[visitor.context.pc].accept(&visitor, ())

                } catch InterpreterInterrupt.spawn(let context) {
                    self.contextBag.add(context)
                }
            } while !visitor.context.terminated

        } catch InterpreterInterrupt.switchPoint {

        } catch {
            throw error
        }

        contextBag.remove(context)
        contextBag.add(visitor.context)
    }

}
