//
//  StatefulModelChecker.swift
//  
//
//  Created by William Ma on 10/31/21.
//

import Foundation

class StatefulModelChecker {

    struct State: Hashable {

        static let initialState = State(
            contextBag: Bag([.initContext]),
            vars: Dict(),
            nextContextToRun: .initContext
        )

        var contextBag: Bag<Context>
        var vars: Dict

        var nextContextToRun: Context
        var nextIndexToChoose: Int?

        var nonterminatedContexts: Swift.Set<Context> {
            contextBag.elements().filter { !$0.terminated }
        }

        var allTerminated: Bool {
            nonterminatedContexts.isEmpty
        }

        var runnable: Swift.Set<Context> {
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
        case switchPoint(Op)
    }

    struct Visitor: DeterministicContextOpVisitor {

        var context: Context
        let modelChecker: StatefulModelChecker

        init(context: Context, modelChecker: StatefulModelChecker) {
            self.context = context
            self.modelChecker = modelChecker
        }

        func choose() throws {
            guard case let .set(s) = context.stack.last else {
                throw OpError.stackTypeMismatch(expected: .set)
            }

            throw Interrupt.choose(s.count)
        }

        func store(address: Value?) throws {
            throw Interrupt.switchPoint(.store(address: address))
        }

        func load(address: Value?) throws {
            throw Interrupt.switchPoint(.load(address: address))
        }

        func spawn(eternal: Bool) throws {
            throw Interrupt.switchPoint(.spawn(eternal: eternal))
        }

        mutating func atomicInc(lazy: Bool) throws {
            if !context.isAtomic {
                try OpImpl.atomicInc(context: &context, lazy: lazy)
                throw Interrupt.switchPoint(.atomicInc(lazy: lazy))
            } else {
                try OpImpl.atomicInc(context: &context, lazy: lazy)
            }
        }

        mutating func atomicDec() throws {
            try OpImpl.atomicDec(context: &context)
            if !context.isAtomic {
                throw Interrupt.switchPoint(.atomicDec)
            }
        }

    }

    let code: [Op]

    init(code: [Op]) {
        self.code = code
    }

    func run() throws {
        var visited: Swift.Set<State> = []
        var stack: [State] = [.initialState]

        while var state = stack.popLast() {
            if visited.contains(state) {
                continue
            }

            visited.insert(state)
            assert(state.contextBag.contains(state.nextContextToRun))
            var visitor = Visitor(context: state.nextContextToRun, modelChecker: self)

            logger.trace("Context switch to \(visitor.context.name)")
            do {
                while !visitor.context.terminated {
                    logger.trace("\(visitor.context.name), \(code[visitor.context.pc]), \(visitor.context.stack)")
                    try code[visitor.context.pc].accept(&visitor)
                }

            } catch let i as Interrupt {
                switch i {
                case .choose(let count):
                    state.contextBag.remove(state.nextContextToRun)

                    for i in 0..<count {
                        var context = visitor.context
                        try OpImpl.choose(context: &context, chooseFn: { _ in i })
                        var newState = state
                        newState.contextBag.add(context)
                        newState.nextContextToRun = context
                        stack.append(newState)
                    }

                case .switchPoint(let op):
                    switch op {
                    case .atomicInc, .atomicDec:
                        break

                    case .spawn(eternal: let eternal):
                        let child = try OpImpl.spawn(parent: &visitor.context, name: "", eternal: eternal)
                        state.spawnCounter += 1
                        state.contextBag.add(child)

                    case .load(address: let address):
                        try OpImpl.load(context: &visitor.context, vars: &state.vars, address: address)

                    case .store(address: let address):
                        try OpImpl.store(context: &visitor.context, vars: &state.vars, address: address)

                    default:
                        fatalError()
                    }

                    state.contextBag.remove(state.nextContextToRun)
                    state.contextBag.add(visitor.context)

                    for context in state.runnable {
                        var newState = state
                        newState.nextContextToRun = context
                        stack.append(newState)
                    }
                }
            }
        }
    }

}