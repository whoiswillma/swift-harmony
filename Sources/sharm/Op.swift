//
//  File.swift
//  
//
//  Created by William Ma on 10/26/21.
//

import Foundation

enum Nary: String, Hashable {

    case minus
    case not
    case equals
    case dictAdd
    case plus
    case atLabel

    var arity: Int {
        switch self {
        case .minus: return 2
        case .not: return 1
        case .equals: return 2
        case .dictAdd: return 3
        case .plus: return 2
        case .atLabel: return 1
        }
    }

}

enum Op: Hashable {

    case frame(name: String, params: [String])
    case push(Value)
    case sequential
    case choose
    case store(Value?)
    case storeAddress
    case storeVar(String)
    case jump(pc: Int)
    case jumpCond(pc: Int, cond: Value)
    case loadVar(String)
    case load(Value?)
    case address
    case nary(Nary)
    case atomicInc(lazy: Bool)
    case atomicDec
    case readonlyInc
    case readonlyDec
    case assertOp
    case delVar(String)
    case ret
    case spawn(eternal: Bool)
    case apply
    case pop

}

extension Op: Decodable {

    enum CodingKeys: String, CodingKey {
        case op
        case name
        case args
        case value
        case pc
        case arity
        case lazy
        case eternal
        case cond
    }

    init(from decoder: Decoder) throws {
        let values = try decoder.container(keyedBy: CodingKeys.self)
        let op = try values.decode(String.self, forKey: .op)

        switch op {
        case "Frame":
            let name = try values.decode(String.self, forKey: .name)
            let params = try values.decode(String.self, forKey: .args)
            if params == "()" {
                self = .frame(name: name, params: [])
            } else if params.starts(with: "(") {
                let afterFirst = params.index(after: params.startIndex)
                let beforeLast = params.index(before: params.endIndex)
                assert(params[beforeLast] == ")")
                let commaSeparated = params[afterFirst..<beforeLast]
                    .split(separator: ",")
                    .map {
                        $0.trimmingCharacters(in: .whitespacesAndNewlines)
                    }
                self = .frame(name: name, params: commaSeparated)
            } else {
                // just one param
                self = .frame(name: name, params: [params])
            }

        case "Jump":
            let pc = Int(try values.decode(String.self, forKey: .pc))!
            self = .jump(pc: pc)

        case "DelVar":
            let value = try values.decode(String.self, forKey: .value)
            self = .delVar(value)

        case "LoadVar":
            let value = try values.decode(String.self, forKey: .value)
            self = .loadVar(value)

        case "Nary":
            let value = try values.decode(String.self, forKey: .value)

            let nary: Nary
            switch value {
            case "+":
                nary = .plus

            case "-":
                nary = .minus

            case "not":
                nary = .not

            case "==":
                nary = .equals

            case "atLabel":
                nary = .atLabel

            case "DictAdd":
                nary = .dictAdd

            default:
                fatalError(value)
            }

            let arity = try values.decode(Int.self, forKey: .arity)
            assert(nary.arity == arity)
            self = .nary(nary)

        case "StoreVar":
            let value = try values.decode(String.self, forKey: .value)
            self = .storeVar(value)

        case "Return":
            self = .ret

        case "Push":
            let value = try values.decode(Value.self, forKey: .value)
            self = .push(value)

        case "Apply":
            self = .apply

        case "Pop":
            self = .pop

        case "Load":
            if values.contains(.value) {
                let values = try values.decode([Value].self, forKey: .value)
                self = .load(.address(values))
            } else {
                self = .load(nil)
            }

        case "Sequential":
            self = .sequential

        case "Store":
            if values.contains(.value) {
                let values = try values.decode([Value].self, forKey: .value)
                self = .store(.address(values))
            } else {
                self = .store(nil)
            }

        case "Choose":
            self = .choose

        case "JumpCond":
            let pc = Int(try values.decode(String.self, forKey: .pc))!
            let cond = try values.decode(Value.self, forKey: .cond)
            self = .jumpCond(pc: pc, cond: cond)

        case "Address":
            self = .address

        case "AtomicInc":
            let lazy = try values.decode(String.self, forKey: .lazy)
            assert(lazy == "True" || lazy == "False")
            self = .atomicInc(lazy: lazy == "True")

        case "AtomicDec":
            self = .atomicDec

        case "ReadonlyInc":
            self = .readonlyInc

        case "ReadonlyDec":
            self = .readonlyDec

        case "Assert":
            self = .assertOp

        case "Spawn":
            let eternal = try values.decode(String.self, forKey: .eternal)
            assert(eternal == "True" || eternal == "False")
            self = .spawn(eternal: eternal == "True")

        default:
            fatalError(op)
        }
    }

}

protocol OpVisitor {

    mutating func frame(name: String, params: [String]) throws
    mutating func push(_ value: Value) throws
    mutating func sequential() throws
    mutating func choose() throws
    mutating func store(_ value: Value?) throws
    mutating func storeAddress() throws
    mutating func storeVar(_ varName: String) throws
    mutating func jump(pc: Int) throws
    mutating func jumpCond(pc: Int, cond: Value) throws
    mutating func loadVar(_ varName: String) throws
    mutating func load(_ value: Value?) throws
    mutating func address() throws
    mutating func nary(_ nary: Nary) throws
    mutating func atomicInc(lazy: Bool) throws
    mutating func atomicDec() throws
    mutating func readonlyInc() throws
    mutating func readonlyDec() throws
    mutating func assertOp() throws
    mutating func delVar(_ varName: String) throws
    mutating func ret() throws
    mutating func spawn(eternal: Bool) throws
    mutating func apply() throws
    mutating func pop() throws

}

//extension OpVisitor {
//
//    mutating func frame(name: String, params: [String]) throws { fatalError() }
//    mutating func push(_ value: Value) throws { fatalError() }
//    mutating func sequential() throws { fatalError() }
//    mutating func choose() throws { fatalError() }
//    mutating func store(_ value: Value?) throws { fatalError() }
//    mutating func storeAddress() throws { fatalError() }
//    mutating func storeVar(_ varName: String) throws { fatalError() }
//    mutating func jump(pc: Int) throws { fatalError() }
//    mutating func jumpCond(pc: Int, cond: Value) throws { fatalError() }
//    mutating func loadVar(_ varName: String) throws { fatalError() }
//    mutating func load(_ value: Value?) throws { fatalError() }
//    mutating func address() throws { fatalError() }
//    mutating func nary(_ nary: Nary) throws { fatalError() }
//    mutating func atomicInc(lazy: Bool) throws { fatalError() }
//    mutating func atomicDec() throws { fatalError() }
//    mutating func readonlyInc() throws { fatalError() }
//    mutating func readonlyDec() throws { fatalError() }
//    mutating func assertOp() throws { fatalError() }
//    mutating func delVar(_ varName: String) throws { fatalError() }
//    mutating func ret() throws { fatalError() }
//    mutating func spawn(eternal: Bool) throws { fatalError() }
//    mutating func apply() throws { fatalError() }
//    mutating func pop() throws { fatalError() }
//
//}

extension Op {

    func accept(_ visitor: inout OpVisitor) throws {
        switch self {
        case .frame(name: let name, params: let params):
            try visitor.frame(name: name, params: params)
        case .push(let value):
            try visitor.push(value)
        case .sequential:
            try visitor.sequential()
        case .choose:
            try visitor.choose()
        case .store(let value):
            try visitor.store(value)
        case .storeAddress:
            try visitor.storeAddress()
        case .storeVar(let varName):
            try visitor.storeVar(varName)
        case .jump(pc: let pc):
            try visitor.jump(pc: pc)
        case .jumpCond(pc: let pc, cond: let cond):
            try visitor.jumpCond(pc: pc, cond: cond)
        case .loadVar(let varName):
            try visitor.loadVar(varName)
        case .load(let value):
            try visitor.load(value)
        case .address:
            try visitor.address()
        case .nary(let nary):
            try visitor.nary(nary)
        case .atomicInc(lazy: let lazy):
            try visitor.atomicInc(lazy: lazy)
        case .atomicDec:
            try visitor.atomicDec()
        case .readonlyInc:
            try visitor.readonlyInc()
        case .readonlyDec:
            try visitor.readonlyDec()
        case .assertOp:
            try visitor.assertOp()
        case .delVar(let varName):
            try visitor.delVar(varName)
        case .ret:
            try visitor.ret()
        case .spawn(eternal: let eternal):
            try visitor.spawn(eternal: eternal)
        case .apply:
            try visitor.apply()
        case .pop:
            try visitor.pop()
        }
    }

}

struct DeterministicContextOpVisitor: OpVisitor {

    var context: Context
    var unimplemented: Bool = false

    mutating func frame(name: String, params: [String]) throws {
        guard case let .dict(args) = context.stack.last! else { fatalError() }

        // save the current vars
        context.stack.append(.dict(context.vars))
        context.stack.append(.int(context.fp))

        // match args with params
        context.vars = Dict()
        assert(args.count == params.count)
        for i in 0..<params.count {
            context.vars[.atom(params[i])] = args.elements[i].value
        }

        context.fp = context.stack.count
        context.pc += 1
    }

    mutating func jump(pc: Int) throws {
        context.pc = pc
    }

    mutating func delVar(_ varName: String) throws {
        context.vars[.atom(varName)] = nil
        context.pc += 1
    }

    mutating func loadVar(_ varName: String) throws {
        let value = context.vars[.atom(varName)]!
        context.stack.append(value)
        context.pc += 1
    }

    mutating func nary(_ nary: Nary) throws {
        switch nary {
        case .plus:
            let rhs = context.stack.popLast()
            let lhs = context.stack.popLast()

            let result: Value
            switch (lhs, rhs) {
            case let (.int(lhs), .int(rhs)):
                result = .int(lhs + rhs)

            default:
                fatalError()
            }
            context.stack.append(result)

        case .minus:
            let rhs = context.stack.popLast()
            let lhs = context.stack.popLast()

            let result: Value
            switch (lhs, rhs) {
            case let (.int(lhs), .int(rhs)):
                result = .int(lhs - rhs)

            default:
                fatalError()
            }
            context.stack.append(result)

        default:
            fatalError()
        }

        context.pc += 1
    }

    mutating func storeVar(_ varName: String) throws {
        let value = context.stack.popLast()!
        context.vars[.atom(varName)] = value
        context.pc += 1
    }

    mutating func ret() throws {
        let result = context.vars[.atom("result")] ?? .noneValue

        guard case let .int(originalFp) = context.stack.popLast()! else { fatalError() }
        context.fp = originalFp

        guard case let .dict(originalVars) = context.stack.popLast()! else { fatalError() }
        context.vars = originalVars

        // pop call arguments
        _ = context.stack.popLast()!

        if context.stack.isEmpty {
            context.terminated = true
            return
        }

        guard case let .pc(returnPc) = context.stack.popLast()! else { fatalError() }
        context.pc = returnPc

        context.stack.append(result)
    }

    mutating func push(_ value: Value) throws {
        let args = context.stack.popLast()!
        let f = context.stack.popLast()!

        switch f {
        case let .dict(dict):
            context.stack.append(dict[args]!)

        case let .pc(pc):
            context.stack.append(.pc(context.pc + 1))
            context.stack.append(args)
            context.pc = pc

        default:
            fatalError()
        }
    }

    mutating func pop() throws {
        _ = context.stack.popLast()!
        context.pc += 1
    }

    mutating func jumpCond(pc: Int, cond: Value) throws {
        let test = context.stack.popLast()!
        if test == cond {
            context.pc = pc
        } else {
            context.pc += 1
        }
    }

    mutating func address() throws {
        let value = context.stack.popLast()!
        guard case var .address(values) = context.stack.popLast()! else { fatalError() }
        values.append(value)
        context.stack.append(.address(values))
    }

    mutating func sequential() throws {
        context.pc += 1
    }

    mutating func choose() throws {
        unimplemented = true
    }

    mutating func store(_ value: Value?) throws {
        unimplemented = true
    }

    mutating func storeAddress() throws {
        unimplemented = true
    }

    mutating func load(_ value: Value?) throws {
        unimplemented = true
    }

    mutating func atomicInc(lazy: Bool) throws {
        unimplemented = true
    }

    mutating func atomicDec() throws {
        unimplemented = true
    }

    mutating func readonlyInc() throws {
        unimplemented = true
    }

    mutating func readonlyDec() throws {
        unimplemented = true
    }

    mutating func assertOp() throws {
        unimplemented = true
    }

    mutating func spawn(eternal: Bool) throws {
        unimplemented = true
    }

    mutating func apply() throws {
        unimplemented = true
    }

}

extension Op {

    func applyDeterministic(_ context: inout Context) -> Bool {
        switch self {
        case let .frame(name: _, params: params):
            guard case let .dict(args) = context.stack.last! else { fatalError() }

            // save the current vars
            context.stack.append(.dict(context.vars))
            context.stack.append(.int(context.fp))

            // match args with params
            context.vars = Dict()
            assert(args.count == params.count)
            for i in 0..<params.count {
                context.vars[.atom(params[i])] = args.elements[i].value
            }

            context.fp = context.stack.count
            context.pc += 1

        case let .jump(pc: pc):
            context.pc = pc

        case let .delVar(varName):
            context.vars[.atom(varName)] = nil
            context.pc += 1

        case let .loadVar(varName):
            let value = context.vars[.atom(varName)]!
            context.stack.append(value)
            context.pc += 1

        case let .nary(nary):
            switch nary {
            case .plus:
                let rhs = context.stack.popLast()
                let lhs = context.stack.popLast()

                let result: Value
                switch (lhs, rhs) {
                case let (.int(lhs), .int(rhs)):
                    result = .int(lhs + rhs)

                default:
                    fatalError()
                }
                context.stack.append(result)

            case .minus:
                let rhs = context.stack.popLast()
                let lhs = context.stack.popLast()

                let result: Value
                switch (lhs, rhs) {
                case let (.int(lhs), .int(rhs)):
                    result = .int(lhs - rhs)

                default:
                    fatalError()
                }
                context.stack.append(result)

            default:
                fatalError()
            }

            context.pc += 1

        case let .storeVar(varName):
            let value = context.stack.popLast()!
            context.vars[.atom(varName)] = value
            context.pc += 1

        case .ret:
            let result = context.vars[.atom("result")] ?? .noneValue

            guard case let .int(originalFp) = context.stack.popLast()! else { fatalError() }
            context.fp = originalFp

            guard case let .dict(originalVars) = context.stack.popLast()! else { fatalError() }
            context.vars = originalVars

            // pop call arguments
            _ = context.stack.popLast()!

            if context.stack.isEmpty {
                context.terminated = true
                break
            }

            guard case let .pc(returnPc) = context.stack.popLast()! else { fatalError() }
            context.pc = returnPc

            context.stack.append(result)

        case let .push(value):
            context.stack.append(value)
            context.pc += 1

        case .apply:
            let args = context.stack.popLast()!
            let f = context.stack.popLast()!

            switch f {
            case let .dict(dict):
                context.stack.append(dict[args]!)

            case let .pc(pc):
                context.stack.append(.pc(context.pc + 1))
                context.stack.append(args)
                context.pc = pc

            default:
                fatalError()
            }

        case .pop:
            _ = context.stack.popLast()!
            context.pc += 1

        case let .jumpCond(pc: pc, cond: value):
            let test = context.stack.popLast()!
            if test == value {
                context.pc = pc
            } else {
                context.pc += 1
            }

        case .address:
            let value = context.stack.popLast()!
            guard case var .address(values) = context.stack.popLast()! else { fatalError() }
            values.append(value)
            context.stack.append(.address(values))

        case .sequential:
            context.pc += 1

        default:
            return false
        }

        return true
    }

    func applyNondeterministic(to context: inout Context, nondeterminism: inout Nondeterminism) -> Bool {
        switch self {
        case .choose:
            guard case let .set(value) = context.stack.popLast()! else { fatalError() }
            let sorted = value.sorted()
            let chosen = sorted[nondeterminism.chooseIndex(sorted)]
            context.stack.append(chosen)

            context.pc += 1

        default:
            return false
        }

        return true
    }

    func applyStateful(to state: inout State) -> Bool {
        switch self {
        case .load(let addressOrNil):
            var context = state.contexts.remove(index: state.current)

            let addresses: [Value]
            if case let .address(addrs) = addressOrNil {
                addresses = addrs
            } else if case let .address(addrs) = context.stack.popLast()! {
                addresses = addrs
            } else {
                fatalError()
            }

            var dict = state.vars
            for address in addresses[..<(addresses.count - 1)] {
                guard case let .dict(d) = dict[address] else { fatalError() }
                dict = d
            }

            let result = dict[addresses.last!]!
            context.stack.append(result)

            context.pc += 1
            state.current = state.contexts.add(context)

        case .store(let addressOrNil):
            var context = state.contexts.remove(index: state.current)

            let value = context.stack.popLast()!

            let addresses: [Value]
            if case let .address(addrs) = addressOrNil {
                addresses = addrs
            } else if case let .address(addrs) = context.stack.popLast()! {
                addresses = addrs
            } else {
                fatalError()
            }

            var dict = state.vars
            dict = dict.replacing(valueAt: addresses, with: value)!
            state.vars = dict

            context.pc += 1
            state.current = state.contexts.add(context)


        default:
            return false
        }

        return true
    }

    func apply(to state: inout State) -> Bool {
        var newContext = state.contexts.get(index: state.current)
        if applyDeterministic(&newContext) {
            _ = state.contexts.remove(index: state.current)
            state.current = state.contexts.add(newContext)
            return true
        } else if applyNondeterministic(to: &newContext, nondeterminism: &state.nondeterminism) {
            _ = state.contexts.remove(index: state.current)
            state.current = state.contexts.add(newContext)
            return true
        } else if applyStateful(to: &state) {
            return true
        }

        return false
    }

}

private extension Dict {

    func replacing(valueAt indexPath: [Value], with value: Value) -> Dict? {
        assert(!indexPath.isEmpty)
        if indexPath.count == 1 {
            var copy = self
            copy[indexPath[0]] = value
            return copy
        } else {
            guard let index = index(forKey: indexPath[0]) else { return nil }
            guard case let .dict(dict) = elements[index].value else { return nil }
            guard let result = dict.replacing(valueAt: Array(indexPath[1...]), with: value) else { return nil }
            var copy = self
            copy[indexPath[0]] = .dict(result)
            return copy
        }
    }

}
