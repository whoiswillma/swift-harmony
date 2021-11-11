//
//  StatefulModelChecker.swift
//  
//
//  Created by William Ma on 10/31/21.
//

class StatefulModelChecker {

    struct State: Hashable {

        static let initialState = State(
            contextBag: Bag([.initContext]),
            vars: HDict(),
            nextContextToRun: .initContext
        )

        var contextBag: Bag<Context>
        var vars: HDict

        var nextContextToRun: Context

        var nonterminatedContexts: Set<Context> {
            contextBag.elements().filter { !$0.terminated }
        }

        var allTerminated: Bool {
            nonterminatedContexts.isEmpty
        }

        var runnable: Set<Context> {
            let contexts = nonterminatedContexts

            if let context = contexts.first(where: { $0.isAtomic }) {
                return [context]
            } else {
                return contexts
            }
        }

    }

    enum Interrupt: Error {
        case choose(Int)
        case switchPoint
    }

    struct Visitor: DefaultOpVisitor {

        var state: State
        var context: Context

        init(state: State) {
            self.state = state
            self.context = state.nextContextToRun
        }

        func choose() throws {
            guard case let .set(s) = context.stack.last else {
                throw OpError.stackTypeMismatch(expected: .set)
            }

            throw Interrupt.choose(s.count)
        }

        mutating func atomicInc(lazy: Bool) throws {
            let switchPoint = !context.isAtomic

            try OpImpl.atomicInc(context: &context, lazy: lazy)

            if switchPoint {
                throw Interrupt.switchPoint
            }
        }

        mutating func atomicDec() throws {
            try OpImpl.atomicDec(context: &context)

            if !context.isAtomic {
                throw Interrupt.switchPoint
            }
        }

        mutating func nary(nary: Nary) throws {
            try OpImpl.nary(context: &context, contextBag: state.contextBag, nary: nary)
        }

        mutating func spawn(eternal: Bool) throws {
            let child = try OpImpl.spawn(parent: &context, name: "", eternal: eternal)
            state.contextBag.add(child)
            throw Interrupt.switchPoint
        }

        mutating func load(address: Value?) throws {
            try OpImpl.load(context: &context, vars: &state.vars, address: address)
            throw Interrupt.switchPoint
        }

        mutating func store(address: Value?) throws {
            try OpImpl.store(context: &context, vars: &state.vars, address: address)
            throw Interrupt.switchPoint
        }

    }

    let code: [Op]

    init(code: [Op]) {
        self.code = code
    }

    func run() throws {
        var visited: Set<State> = []
        var boundary: Set<State> = [.initialState]

        while var state = boundary.popFirst() {
            if visited.contains(state) {
                continue
            }

            visited.insert(state)
            assert(state.contextBag.contains(state.nextContextToRun))
            var visitor = Visitor(state: state)

            logger.trace("Context switch to \(visitor.context.name)")
            do {
                while !visitor.context.terminated {
                    logger.trace("  \(code[visitor.context.pc]), \(visitor.context)")
                    try code[visitor.context.pc].accept(&visitor)
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
                        boundary.insert(newState)
                    }

                case .switchPoint:
                    state.contextBag.remove(state.nextContextToRun)
                    state.contextBag.add(visitor.context)

                    for context in state.runnable {
                        var newState = state
                        newState.nextContextToRun = context
                        boundary.insert(newState)
                    }
                }
            }
        }
    }

}
