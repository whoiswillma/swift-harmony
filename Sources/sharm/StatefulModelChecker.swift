//
//  StatefulModelChecker.swift
//  
//
//  Created by William Ma on 10/31/21.
//

class StatefulModelChecker {

    enum Interrupt: Error {
        case choose(Int)
        case switchPoint
    }

    struct Visitor: DefaultOpImplVisitor {

        typealias Input = Void
        typealias Output = Void

        var state: State
        var context: Context

        init(state: State) {
            self.state = state
            self.context = state.nextContextToRun
        }

        func choose(_ input: Void) throws {
            guard case let .set(s) = context.stack.last else {
                throw OpError.stackTypeMismatch(expected: .set)
            }

            throw Interrupt.choose(s.count)
        }

        mutating func atomicInc(lazy: Bool, _ input: Void) throws {
            let switchPoint = !context.isAtomic

            try OpImpl.atomicInc(context: &context, lazy: lazy)

            if switchPoint {
                throw Interrupt.switchPoint
            }
        }

        mutating func atomicDec(_ input: Void) throws {
            try OpImpl.atomicDec(context: &context)

            if !context.isAtomic {
                throw Interrupt.switchPoint
            }
        }

        mutating func nary(nary: Nary, _ input: Void) throws {
            try OpImpl.nary(context: &context, contextBag: state.contextBag, nary: nary)
        }

        mutating func spawn(eternal: Bool, _ input: Void) throws {
            let child = try OpImpl.spawn(parent: &context, name: "", eternal: eternal)
            state.contextBag.add(child)
            if !context.isAtomic {
                throw Interrupt.switchPoint
            }
        }

        mutating func load(address: Value?, _ input: Void) throws {
            try OpImpl.load(context: &context, vars: &state.vars, address: address)
            if !context.isAtomic {
                throw Interrupt.switchPoint
            }
        }

        mutating func store(address: Value?, _ input: Void) throws {
            try OpImpl.store(context: &context, vars: &state.vars, address: address)
            if !context.isAtomic {
                throw Interrupt.switchPoint
            }
        }

    }

    let code: [Op]

    init(code: [Op]) {
        self.code = code
    }

    func run() throws {
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
            assert(state.contextBag.contains(state.nextContextToRun))
            var visitor = Visitor(state: state)

            logger.trace("Context switch to \(visitor.context.name)")
            do {
                while !visitor.context.terminated {
                    logger.trace("  \(code[visitor.context.pc]), \(visitor.context)")
                    try code[visitor.context.pc].accept(&visitor, ())
                }

                throw Interrupt.switchPoint

            } catch let i as Interrupt {
                state = visitor.state

                switch i {
                case .choose(let count):
                    state.contextBag.remove(state.nextContextToRun)

                    for i in 0..<count {
                        var context = visitor.context
                        try OpImpl.choose(context: &context, chooseFn: { _ in i })
                        var newState = state
                        newState.contextBag.add(context)
                        newState.nextContextToRun = context
                        boundary.append(newState)
                    }

                case .switchPoint:
                    state.contextBag.remove(state.nextContextToRun)
                    state.contextBag.add(visitor.context)

                    for context in state.runnable {
                        var newState = state
                        newState.nextContextToRun = context
                        boundary.append(newState)
                    }
                }
            }
        }
    }

}
