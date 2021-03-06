//
//  OpImpl.swift
//  
//
//  Created by William Ma on 11/12/21.
//

import Foundation

enum OpImpl {

    private static func matchVarTree(varTree: VarTree, value: Value, vars: inout HDict) throws {
        switch varTree {
        case .name("_"):
            break

        case .name(let name):
            vars[.atom(name)] = value

        case .tuple(let elems):
            guard case let .dict(d) = value else {
                throw OpError.typeMismatch(expected: [.dict], actual: [value.type])
            }

            assert(d.count == elems.count)

            for (elem, (_, value)) in zip(elems, d) {
                try matchVarTree(varTree: elem, value: value, vars: &vars)
            }
        }
    }

    static func frame(context: inout Context, name: String, params: VarTree) throws {
        guard let args = context.stack.last else {
            throw OpError.stackIsEmpty
        }

        // save the current vars
        context.stack.append(.dict(context.vars))

        let valueThis = context.vars[.atom("this")]
        context.vars = HDict()
        context.vars[.atom("this")] = valueThis

        try matchVarTree(varTree: params, value: args, vars: &context.vars)

        context.pc += 1
    }

    static func jump(context: inout Context, pc: Int) throws {
        context.pc = pc
    }

    static func delVar(context: inout Context, varName: String?) throws {
        if let varName = varName {
            context.vars[.atom(varName)] = nil

        } else {
            guard case let .address(indexPath) = context.stack.popLast() else {
                throw OpError.stackTypeMismatch(expected: .address)
            }

            try context.vars.replace(valueAt: indexPath, with: nil)
        }

        context.pc += 1
    }

    static func loadVar(context: inout Context, varName: String?) throws {
        if let varName = varName {
            guard let value = context.vars[.atom(varName)] else {
                throw OpError.unknownVar(varName: varName)
            }
            context.stack.append(value)

        } else {
            guard case let .address(indexPath) = context.stack.popLast() else {
                throw OpError.stackTypeMismatch(expected: .address)
            }

            guard let value = context.vars.value(at: indexPath) else {
                throw OpError.invalidAddress(address: .address(indexPath))
            }

            context.stack.append(value)
        }

        context.pc += 1
    }

    static func nary(context: inout Context, nary n: Nary) throws {
        switch n {
        case .plus(arity: let arity):
            try NaryImpl.plus(context: &context, arity: arity)

        case .minus:
            try NaryImpl.minus(context: &context)

        case .not:
            try NaryImpl.not(context: &context)

        case .dictAdd:
            try NaryImpl.dictAdd(context: &context)

        case .equals:
            try NaryImpl.equals(context: &context)

        case .range:
            try NaryImpl.range(context: &context)

        case .isEmpty:
            try NaryImpl.isEmpty(context: &context)

        case .len:
            try NaryImpl.len(context: &context)

        case .times(arity: let arity):
            try NaryImpl.times(context: &context, arity: arity)

        case .mod:
            try NaryImpl.mod(context: &context)

        case .notEquals:
            try NaryImpl.notEquals(context: &context)

        case .min:
            try NaryImpl.min(context: &context)

        case .max:
            try NaryImpl.max(context: &context)

        case .lessThan:
            try NaryImpl.lessThan(context: &context)

        case .greaterThan:
            try NaryImpl.greaterThan(context: &context)

        case .lessThanOrEqual:
            try NaryImpl.lessThanOrEqual(context: &context)

        case .greaterThanOrEqual:
            try NaryImpl.greaterThanOrEqual(context: &context)

        case .setAdd:
            try NaryImpl.setAdd(context: &context)

        case .keys:
            try NaryImpl.keys(context: &context)

        case .divide:
            try NaryImpl.divide(context: &context)

        default:
            throw OpError.unimplemented("Nary \(n)")
        }

        context.pc += 1
    }

    static func nary(context: inout Context, contextArray: [Context], nary n: Nary) throws {
        switch n {
        case .atLabel:
            try NaryImpl.atLabel(context: &context, contextArray: contextArray)

        default:
            try nary(context: &context, nary: n)
        }
    }

    static func nary(context: inout Context, contextBag: Bag<Context>, nary n: Nary) throws {
        switch n {
        case .atLabel:
            try NaryImpl.atLabel(context: &context, contextBag: contextBag)

        default:
            try nary(context: &context, nary: n)
        }
    }

    static func storeVar(context: inout Context, varTree: VarTree?) throws {
        guard let value = context.stack.popLast() else {
            throw OpError.stackIsEmpty
        }

        if let varTree = varTree {
            try matchVarTree(varTree: varTree, value: value, vars: &context.vars)

        } else {
            guard case let .address(indexPath) = context.stack.popLast() else {
                throw OpError.stackTypeMismatch(expected: .address)
            }

            try context.vars.replace(valueAt: indexPath, with: value)
        }

        context.pc += 1
    }

    static func ret(context: inout Context) throws {
        let result = context.vars[.atom("result")] ?? .noneValue

        guard case let .dict(originalVars) = context.stack.popLast() else {
            throw OpError.stackTypeMismatch(expected: .dict)
        }
        let thisValue = context.vars[.atom("this")]
        context.vars = originalVars
        context.vars[.atom("this")] = thisValue

        // pop call arguments
        guard nil != context.stack.popLast() else {
            throw OpError.stackIsEmpty
        }

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

    static func push(context: inout Context, value: Value) throws {
        context.stack.append(value)
        context.pc += 1
    }

    static func pop(context: inout Context) throws {
        guard nil != context.stack.popLast() else {
            throw OpError.stackIsEmpty
        }

        context.pc += 1
    }

    static func jumpCond(context: inout Context, pc: Int, cond: Value) throws {
        guard let test = context.stack.popLast() else {
            throw OpError.stackIsEmpty
        }

        if test == cond {
            context.pc = pc
        } else {
            context.pc += 1
        }
    }

    static func address(context: inout Context) throws {
        guard let value = context.stack.popLast() else {
            throw OpError.stackIsEmpty
        }

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

    static func choose(context: inout Context, chooseFn: (HSet) throws -> Int) throws {
        guard case let .set(value) = context.stack.popLast() else {
            throw OpError.stackTypeMismatch(expected: .set)
        }

        let chosen = value[value.index(value.startIndex, offsetBy: try chooseFn(value))]
        context.stack.append(chosen)

        context.pc += 1
    }

    static func load(
        context: inout Context,
        vars: inout HDict,
        address: Value?
    ) throws {
        let indexPath: [Value]

        if case let .address(addrs) = address {
            indexPath = addrs
        } else {
            guard case let .address(addrs) = context.stack.popLast() else {
                throw OpError.stackTypeMismatch(expected: .address)
            }

            indexPath = addrs
        }

        guard let result = vars.value(at: indexPath) else {
            throw OpError.invalidAddress(address: .address(indexPath))
        }
        context.stack.append(result)

        context.pc += 1
    }

    static func store(
        context: inout Context,
        vars: inout HDict,
        address: Value?
    ) throws {
        if context.isReadonly { throw OpError.contextIsReadonly }
        guard let value = context.stack.popLast() else { throw OpError.stackIsEmpty }

        let indexPath: [Value]

        if case let .address(addrs) = address {
            indexPath = addrs
        } else {
            guard case let .address(addrs) = context.stack.popLast() else {
                throw OpError.stackTypeMismatch(expected: .address)
            }

            indexPath = addrs
        }

        try vars.replace(valueAt: indexPath, with: value)

        context.pc += 1
    }

    static func apply(context: inout Context) throws {
        guard let args = context.stack.popLast(),
              let f = context.stack.popLast()
        else {
            throw OpError.stackIsEmpty
        }

        switch f {
        case let .dict(dict):
            guard let result = dict[args] else {
                throw OpError.invalidKey(key: args)
            }

            context.stack.append(result)
            context.pc += 1

        case let .pc(pc):
            context.stack.append(.pc(context.pc + 1))
            context.stack.append(.int(Calltype.normal.rawValue))
            context.stack.append(args)
            context.pc = pc

        default:
            throw OpError.typeMismatch(expected: [.dict, .pc], actual: [f.type])
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

        if context.atomicLevel == 0 {
            context.atomicPc = context.pc
        }
        context.atomicLevel += 1
        if !lazy {
            context.isAtomic = true
        }

        context.pc += 1
    }

    static func atomicDec(context: inout Context) throws {
        context.atomicLevel -= 1

        if context.atomicLevel == 0 {
            context.atomicPc = -1
            context.isAtomic = false
        }

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

        var context = Context(
            name: name,
            entry: pc,
            arg: arg,
            stack: [.int(Calltype.process.rawValue), arg]
        )
        context.vars[.atom("this")] = this
        return context
    }

    static func cut(context: inout Context, setName: String, key keyTree: VarTree?, value valueTree: VarTree) throws {
        guard let value = context.vars[.atom(setName)] else {
            throw OpError.unknownVar(varName: setName)
        }

        switch value {
        case var .set(set):
            guard let min = set.min() else {
                throw OpError.setIsEmpty
            }

            set.remove(min)
            context.vars[.atom(setName)] = .set(set)
            try matchVarTree(varTree: valueTree, value: min, vars: &context.vars)

        case var .dict(dict):
            guard let (minKey, minValue) = dict.first else {
                throw OpError.dictIsEmpty
            }

            dict.removeFirst()

            context.vars[.atom(setName)] = .dict(dict)
            try matchVarTree(varTree: valueTree, value: minValue, vars: &context.vars)
            if let keyTree = keyTree {
                try matchVarTree(varTree: keyTree, value: minKey, vars: &context.vars)
            }

        default:
            throw OpError.typeMismatch(expected: [.set, .dict], actual: [value.type])
        }

        context.pc += 1
    }

    static func incVar(context: inout Context, varName: String) throws {
        guard case let .int(i) = context.vars[.atom(varName)] else {
            throw OpError.varTypeMismatch(varName: varName, expected: .int)
        }

        context.vars[.atom(varName)] = .int(i + 1)
        context.pc += 1
    }

    static func dup(context: inout Context) throws {
        guard let value = context.stack.last else {
            throw OpError.stackIsEmpty
        }

        context.stack.append(value)
    }

    static func split(context: inout Context, count: Int) throws {
        guard let value = context.stack.popLast() else {
            throw OpError.stackIsEmpty
        }

        switch value {
        case .dict(let dict):
            guard dict.count == count else {
                throw OpError.wrongCountSplit(actual: dict.count, expected: count)
            }

            for i in dict {
                context.stack.append(i.value)
            }

        case .set(let set):
            guard set.count == count else {
                throw OpError.wrongCountSplit(actual: set.count, expected: count)
            }

            for i in set {
                context.stack.append(i)
            }

        default:
            throw OpError.typeMismatch(expected: [.dict, .set], actual: [value.type])
        }

        context.pc += 1
    }

    static func move(context: inout Context, offset: Int) throws {
        context.stack.swapAt(context.stack.count - 1, context.stack.count - offset)

        context.pc += 1
    }

    static func log(context: inout Context) throws {
        guard let _ = context.stack.popLast() else {
            throw OpError.stackIsEmpty
        }

        context.pc += 1
    }

}

private extension HDict {

    func replacing(valueAt indexPath: [Value], with value: Value?) -> HDict? {
        assert(!indexPath.isEmpty)

        if indexPath.count == 1 {
            var copy = self
            copy[indexPath[0]] = value
            return copy
        } else {
            guard let index = index(forKey: indexPath[0]) else { return nil }
            guard case let .dict(dict) = self[index].value else { return nil }
            guard let result = dict.replacing(valueAt: Array(indexPath[1...]), with: value) else { return nil }
            var copy = self
            copy.values[index] = .dict(result)
            return copy
        }
    }

    mutating func replace(valueAt indexPath: [Value], with value: Value?) throws {
        assert(!indexPath.isEmpty)

        if indexPath.count == 1 {
            self[indexPath[0]] = value
        } else {
            guard let index = index(forKey: indexPath[0]) else {
                throw OpError.invalidKey(key: indexPath[0])
            }
            guard case var .dict(dict) = values[index] else {
                throw OpError.typeMismatch(expected: [.dict], actual: [values[index].type])
            }

            try dict.replace(valueAt: Array(indexPath[1...]), with: value)

            values[index] = .dict(dict)
        }
    }

    func value(at indexPath: [Value]) -> Value? {
        var dict = self

        for address in indexPath[..<(indexPath.count - 1)] {
            guard case let .dict(d) = dict[address] else {
                return nil
            }

            dict = d
        }

        guard let last = indexPath.last else {
            return nil
        }

        return dict[last]
    }

}
