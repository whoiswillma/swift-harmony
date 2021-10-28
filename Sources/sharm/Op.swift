//
//  File.swift
//  
//
//  Created by William Ma on 10/26/21.
//

import Foundation

enum OpError: Error {

    case typeMismatch(expected: Swift.Set<ValueType>, actual: [ValueType])
    case assertionFailure
    case unimplemented
    case stackIsEmpty
    case contextIsReadonly
    case contextIsAtomic
    case contextIsNotAtomic
    case stackTypeMismatch(expected: ValueType)
    case invalidAddress(address: Value)
    case invalidCalltype(Int)

}

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

enum NaryImpl {

    static func plus(context: inout Context) throws {
        guard let rhs = context.stack.popLast(),
              let lhs = context.stack.popLast()
        else {
            throw OpError.stackIsEmpty
        }

        let result: Value
        switch (lhs, rhs) {
        case let (.int(lhs), .int(rhs)):
            result = .int(lhs + rhs)

        default:
            throw OpError.typeMismatch(expected: [.int], actual: [lhs.type, rhs.type])
        }

        context.stack.append(result)
    }

    static func minus(context: inout Context) throws {
        guard let rhs = context.stack.popLast(),
              let lhs = context.stack.popLast()
        else {
            throw OpError.stackIsEmpty
        }

        let result: Value
        switch (lhs, rhs) {
        case let (.int(lhs), .int(rhs)):
            result = .int(lhs - rhs)

        default:
            throw OpError.typeMismatch(expected: [.int], actual: [lhs.type, rhs.type])
        }

        context.stack.append(result)
    }

    static func not(context: inout Context) throws {
        guard let op = context.stack.popLast() else { throw OpError.stackIsEmpty }

        let result: Value
        switch op {
        case let .bool(b):
            result = .bool(!b)

        default:
            throw OpError.typeMismatch(expected: [.bool], actual: [op.type])
        }

        context.stack.append(result)
    }

    static func atLabel(context: inout Context, contextBag: Bag<Context>) throws {
        struct ResultKey: Hashable {
            let entry: Int
            let arg: Value
        }

        guard context.isAtomic else {
            throw OpError.contextIsNotAtomic
        }

        guard case let .pc(pc) = context.stack.popLast() else {
            throw OpError.stackTypeMismatch(expected: .pc)
        }

        var result = [ResultKey: Int]()
        for (context, count) in contextBag.elementsWithCount() {
            if context.pc == pc {
                result[ResultKey(entry: context.entry, arg: context.arg), default: 0] += count
            }
        }

        let value = Value.dict(Dict(uniqueKeysWithValues: result.map({ resultKey, count in
            (.dict([.int(0): .pc(resultKey.entry), .int(1): resultKey.arg]), .int(count))
        })))

        context.stack.append(value)
        context.pc += 1
    }

    static func dictAdd(context: inout Context) throws {
        guard let value = context.stack.popLast(),
              let key = context.stack.popLast(),
              let dict = context.stack.popLast()
        else {
            throw OpError.stackIsEmpty
        }

        guard case var .dict(dict) = dict else {
            throw OpError.typeMismatch(expected: [.dict], actual: [dict.type])
        }

        if let existingValue = dict[key] {
            dict[key] = max(value, existingValue)
        } else {
            dict[key] = value
        }

        context.stack.append(.dict(dict))
    }

    static func equals(context: inout Context) throws {
        guard let lhs = context.stack.popLast(),
              let rhs = context.stack.popLast()
        else {
            throw OpError.stackIsEmpty
        }

        context.stack.append(.bool(lhs == rhs))
    }

}

enum Op: Hashable {

    case frame(name: String, params: [String])
    case push(Value)
    case sequential
    case choose
    case store(Value?)
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

extension Op {

    func accept<T: OpVisitor>(_ visitor: inout T) throws {
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

enum OpImpl {

    static func frame(context: inout Context, name: String, params: [String]) throws {
        guard let args = context.stack.last else {
            throw OpError.stackIsEmpty
        }

        // save the current vars
        context.stack.append(.dict(context.vars))
        context.stack.append(.int(context.fp))

        context.vars = Dict()
        switch args {
        case let .dict(args):
            // match args with params
            assert(args.count == params.count)
            for i in 0..<params.count {
                context.vars[.atom(params[i])] = args.elements[i].value
            }

        default:
            assert(params.count == 1)
            context.vars[.atom(params[0])] = args

        }

        context.fp = context.stack.count
        context.pc += 1
    }

    static func jump(context: inout Context, pc: Int) throws {
        context.pc = pc
    }

    static func delVar(context: inout Context, _ varName: String) throws {
        context.vars[.atom(varName)] = nil
        context.pc += 1
    }

    static func loadVar(context: inout Context, _ varName: String) throws {
        let value = context.vars[.atom(varName)]!
        context.stack.append(value)
        context.pc += 1
    }

    static func nary(context: inout Context, _ n: Nary) throws {
        switch n {
        case .plus:
            try NaryImpl.plus(context: &context)

        case .minus:
            try NaryImpl.minus(context: &context)

        case .not:
            try NaryImpl.not(context: &context)

        case .dictAdd:
            try NaryImpl.dictAdd(context: &context)

        case .equals:
            try NaryImpl.equals(context: &context)

        default:
            throw OpError.unimplemented
        }

        context.pc += 1
    }

    static func nary(context: inout Context, contextBag: Bag<Context>, _ n: Nary) throws {
        switch n {
        case .atLabel:
            try NaryImpl.atLabel(context: &context, contextBag: contextBag)

        default:
            try nary(context: &context, n)
        }
    }

    static func storeVar(context: inout Context, _ varName: String) throws {
        let value = context.stack.popLast()!
        context.vars[.atom(varName)] = value
        context.pc += 1
    }

    static func ret(context: inout Context) throws {
        let result = context.vars[.atom("result")] ?? .noneValue

        guard case let .int(originalFp) = context.stack.popLast() else {
            throw OpError.stackTypeMismatch(expected: .int)
        }
        context.fp = originalFp

        guard case let .dict(originalVars) = context.stack.popLast() else {
            throw OpError.stackTypeMismatch(expected: .dict)
        }
        context.vars = originalVars

        // pop call arguments
        _ = context.stack.popLast()!

        if context.stack.isEmpty {
            context.terminated = true
            return
        }

        guard case let .int(calltype) = context.stack.popLast() else {
            throw OpError.stackTypeMismatch(expected: .int)
        }
        guard let calltype = Calltype(rawValue: calltype) else {
            throw OpError.invalidCalltype(calltype)
        }

        switch calltype {
        case .normal:
            guard case let .pc(returnPc) = context.stack.popLast() else {
                throw OpError.stackTypeMismatch(expected: .pc)
            }
            context.pc = returnPc

            context.stack.append(result)

        case .process:
            context.terminated = true
        }
    }

    static func push(context: inout Context, _ value: Value) throws {
        context.stack.append(value)
        context.pc += 1
    }

    static func pop(context: inout Context) throws {
        _ = context.stack.popLast()!
        context.pc += 1
    }

    static func jumpCond(context: inout Context, pc: Int, cond: Value) throws {
        let test = context.stack.popLast()!
        if test == cond {
            context.pc = pc
        } else {
            context.pc += 1
        }
    }

    static func address(context: inout Context) throws {
        let value = context.stack.popLast()!
        guard case var .address(values) = context.stack.popLast() else {
            throw OpError.stackTypeMismatch(expected: .address)
        }
        values.append(value)
        context.stack.append(.address(values))
        context.pc += 1
    }

    static func sequential(context: inout Context) throws {
        guard case .address = context.stack.popLast() else {
            throw OpError.stackTypeMismatch(expected: .address)
        }
        context.pc += 1
    }

    static func choose(context: inout Context, nondeterminism: Nondeterminism) throws {
        guard case let .set(value) = context.stack.popLast()! else { fatalError() }
        let chosen = value.elements[nondeterminism.chooseIndex(value)]
        context.stack.append(chosen)

        context.pc += 1
    }

    static func load(
        context: inout Context,
        vars: inout Dict,
        _ addressOrNil: Value?
    ) throws {
        let addresses: [Value]
        if case let .address(addrs) = addressOrNil {
            addresses = addrs
        } else if case let .address(addrs) = context.stack.popLast()! {
            addresses = addrs
        } else {
            fatalError()
        }

        var dict = vars
        for address in addresses[..<(addresses.count - 1)] {
            guard case let .dict(d) = dict[address] else { fatalError() }
            dict = d
        }

        let result = dict[addresses.last!]!
        context.stack.append(result)

        context.pc += 1
    }

    static func store(
        context: inout Context,
        vars: inout Dict,
        _ addressOrNil: Value?
    ) throws {
        if context.isReadonly { throw OpError.contextIsReadonly }
        guard let value = context.stack.popLast() else { throw OpError.stackIsEmpty }

        let indexPath: [Value]
        if case let .address(addrs) = addressOrNil {
            indexPath = addrs
        } else if case let .address(addrs) = context.stack.popLast()! {
            indexPath = addrs
        } else {
            fatalError()
        }

        guard let result = vars.replacing(valueAt: indexPath, with: value) else {
            throw OpError.invalidAddress(address: .address(indexPath))
        }

        vars = result

        context.pc += 1
    }

    static func apply(context: inout Context) throws {
        let args = context.stack.popLast()!
        let f = context.stack.popLast()!

        switch f {
        case let .dict(dict):
            context.stack.append(dict[args]!)
            context.pc += 1

        case let .pc(pc):
            context.stack.append(.pc(context.pc + 1))
            context.stack.append(args)
            context.pc = pc

        default:
            fatalError()
        }
    }

    static func readonlyInc(context: inout Context) throws {
        assert(context.readonlyLevel >= 0)
        context.readonlyLevel += 1
        context.pc += 1
    }

    static func readonlyDec(context: inout Context) throws {
        context.readonlyLevel -= 1
        assert(context.readonlyLevel >= 0)
        context.pc += 1
    }

    static func assertOp(context: inout Context) throws {
        guard case let .bool(b) = context.stack.popLast() else {
            throw OpError.stackTypeMismatch(expected: .bool)
        }

        if !b {
            throw OpError.assertionFailure
        }

        context.pc += 1
    }

    static func atomicInc(context: inout Context, lazy: Bool) throws {
        assert(context.atomicLevel >= 0)
        context.atomicLevel += 1
        context.pc += 1
    }

    static func atomicDec(context: inout Context) throws {
        context.atomicLevel -= 1
        assert(context.atomicLevel >= 0)
        context.pc += 1
    }

    static func spawn(parent: inout Context, name: String, eternal: Bool) throws -> Context {
        guard let this = parent.stack.popLast(),
              let arg = parent.stack.popLast(),
              let pc = parent.stack.popLast()
        else {
            throw OpError.stackIsEmpty
        }

        guard case let .pc(pc) = pc else {
            throw OpError.typeMismatch(expected: [.pc], actual: [.pc])
        }

        parent.pc += 1
        return Context(
            name: name,
            entry: pc,
            arg: arg,
            stack: [.int(Calltype.process.rawValue), arg]
        )
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

protocol DeterministicContextOpVisitor: OpVisitor {

    var context: Context { get set }

}

extension DeterministicContextOpVisitor {

    mutating func frame(name: String, params: [String]) throws {
        try OpImpl.frame(context: &context, name: name, params: params)
    }

    mutating func jump(pc: Int) throws {
        try OpImpl.jump(context: &context, pc: pc)
    }

    mutating func delVar(_ varName: String) throws {
        try OpImpl.delVar(context: &context, varName)
    }

    mutating func loadVar(_ varName: String) throws {
        try OpImpl.loadVar(context: &context, varName)
    }

    mutating func nary(_ nary: Nary) throws {
        try OpImpl.nary(context: &context, nary)
    }

    mutating func storeVar(_ varName: String) throws {
        try OpImpl.storeVar(context: &context, varName)
    }

    mutating func ret() throws {
        try OpImpl.ret(context: &context)
    }

    mutating func push(_ value: Value) throws {
        try OpImpl.push(context: &context, value)
    }

    mutating func pop() throws {
        try OpImpl.pop(context: &context)
    }

    mutating func jumpCond(pc: Int, cond: Value) throws {
        try OpImpl.jumpCond(context: &context, pc: pc, cond: cond)
    }

    mutating func address() throws {
        try OpImpl.address(context: &context)
    }

    mutating func sequential() throws {
        try OpImpl.sequential(context: &context)
    }

    mutating func apply() throws {
        try OpImpl.apply(context: &context)
    }

    mutating func readonlyInc() throws {
        try OpImpl.readonlyInc(context: &context)
    }

    mutating func readonlyDec() throws {
        try OpImpl.readonlyDec(context: &context)
    }

    mutating func atomicInc(lazy: Bool) throws {
        try OpImpl.atomicInc(context: &context, lazy: lazy)
    }

    mutating func atomicDec() throws {
        try OpImpl.atomicDec(context: &context)
    }

    mutating func assertOp() throws {
        try OpImpl.assertOp(context: &context)
    }

}
