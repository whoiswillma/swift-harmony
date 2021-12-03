//
//  Nary.swift
//  
//
//  Created by William Ma on 10/29/21.
//

enum Nary: Hashable {

    case minus
    case negate
    case not
    case equals
    case notEquals
    case dictAdd
    case plus(arity: Int)
    case atLabel
    case range
    case isEmpty
    case len
    case setAdd
    case keys
    case lessThan
    case greaterThan
    case lessThanOrEqual
    case greaterThanOrEqual
    case `in`
    case min
    case max
    case union(arity: Int)
    case getContext
    case times(arity: Int)
    case mod

    var arity: Int {
        switch self {
        case .minus: return 2
        case .negate: return 1
        case .not: return 1
        case .equals: return 2
        case .dictAdd: return 3
        case .plus(arity: let arity): return arity
        case .atLabel: return 1
        case .range: return 2
        case .isEmpty: return 1
        case .len: return 1
        case .setAdd: return 2
        case .keys: return 1
        case .lessThan: return 2
        case .greaterThanOrEqual: return 2
        case .in: return 2
        case .min: return 1
        case .max: return 1
        case .greaterThan: return 2
        case .lessThanOrEqual: return 2
        case .union(arity: let arity): return arity
        case .getContext: return 1
        case .notEquals: return 2
        case .times(arity: let arity): return arity
        case .mod: return 2
        }
    }

}

enum NaryImpl {

    private static func plusBinary(context: inout Context) throws {
        guard let rhs = context.stack.popLast(),
              let lhs = context.stack.popLast()
        else {
            throw OpError.stackIsEmpty
        }

        let result: Value
        switch (lhs, rhs) {
        case let (.int(lhs), .int(rhs)):
            result = .int(lhs + rhs)

        case let (.dict(lhs), .dict(rhs)):
            let values: [Value] = Array(lhs.values) + Array(rhs.values)
            result = .dict(SortedDictionary(keysWithValues: values.enumerated().map {
                (key: .int($0), value: $1)
            }))

        default:
            throw OpError.typeMismatch(expected: [.int], actual: [lhs.type, rhs.type])
        }

        context.stack.append(result)
    }

    static func plus(context: inout Context, arity: Int) throws {
        for _ in 1..<arity {
            try plusBinary(context: &context)
        }
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

        case let (.set(lhs), .set(rhs)):
            var newSet = lhs
            for elem in newSet {
                if rhs.contains(elem) {
                    newSet.remove(elem)
                }
            }
            result = .set(newSet)

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

    static func atLabel(context: inout Context, contextArray: [Context]) throws {
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
        for context in contextArray {
            if context.atomicPc == pc {
                result[ResultKey(entry: context.entry, arg: context.arg), default: 0] += 1
            }
        }

        let value = Value.dict(HDict(keysWithValues: result.map({ resultKey, count in
            (.dict([.int(0): .pc(resultKey.entry), .int(1): resultKey.arg]), .int(count))
        })))

        context.stack.append(value)
        context.pc += 1
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
        for (other, count) in contextBag.elementsWithCount() {
            if other.atomicPc == pc {
                result[ResultKey(entry: context.entry, arg: context.arg), default: 0] += count
            }
        }

        let value: Value = .dict(HDict(keysWithValues: result.map({ resultKey, count in
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
            dict[key] = Swift.max(value, existingValue)
        } else {
            dict[key] = value
        }

        context.stack.append(.dict(dict))
    }

    static func equals(context: inout Context) throws {
        guard let rhs = context.stack.popLast(),
              let lhs = context.stack.popLast()
        else {
            throw OpError.stackIsEmpty
        }

        context.stack.append(.bool(lhs == rhs))
    }

    static func lessThan(context: inout Context) throws {
        guard let rhs = context.stack.popLast(),
              let lhs = context.stack.popLast()
        else {
            throw OpError.stackIsEmpty
        }

        context.stack.append(.bool(lhs < rhs))
    }

    static func greaterThan(context: inout Context) throws {
        guard let rhs = context.stack.popLast(),
              let lhs = context.stack.popLast()
        else {
            throw OpError.stackIsEmpty
        }

        context.stack.append(.bool(lhs > rhs))
    }

    static func lessThanOrEqual(context: inout Context) throws {
        guard let rhs = context.stack.popLast(),
              let lhs = context.stack.popLast()
        else {
            throw OpError.stackIsEmpty
        }

        context.stack.append(.bool(lhs <= rhs))
    }

    static func greaterThanOrEqual(context: inout Context) throws {
        guard let rhs = context.stack.popLast(),
              let lhs = context.stack.popLast()
        else {
            throw OpError.stackIsEmpty
        }

        context.stack.append(.bool(lhs >= rhs))
    }

    static func range(context: inout Context) throws {
        guard case let .int(rhs) = context.stack.popLast(),
              case let .int(lhs) = context.stack.popLast()
        else {
            throw OpError.stackTypeMismatch(expected: .int)
        }

        if lhs > rhs {
            context.stack.append(.set([]))
        } else {
            let set = HSet((lhs...rhs).map { .int($0) })
            context.stack.append(.set(set))
        }
    }

    static func isEmpty(context: inout Context) throws {
        guard let value = context.stack.popLast() else {
            throw OpError.stackIsEmpty
        }

        let result: Bool
        switch value {
        case let .dict(dict):
            result = dict.isEmpty

        case let .set(set):
            result = set.isEmpty

        default:
            throw OpError.typeMismatch(expected: [.dict, .set], actual: [value.type])
        }

        context.stack.append(.bool(result))
    }

    static func len(context: inout Context) throws {
        guard let value = context.stack.popLast() else {
            throw OpError.stackIsEmpty
        }

        let result: Int
        switch value {
        case .set(let s):
            result = s.count

        case .dict(let d):
            result = d.count

        default:
            throw OpError.typeMismatch(expected: [.dict, .set], actual: [value.type])
        }

        context.stack.append(.int(result))
    }

    static func setAdd(context: inout Context) throws {
        guard let value = context.stack.popLast(),
              let set = context.stack.popLast()
        else {
            throw OpError.stackIsEmpty
        }

        guard case .set(var set) = set else {
            throw OpError.typeMismatch(expected: [.set], actual: [set.type])
        }

        if set.contains(value) {
            return
        } else {
            set.insert(value)
        }

        context.stack.append(.set(set))
    }

    static func times(context: inout Context, arity: Int) throws {
        assert(arity >= 1)

        var list: HDict?
        var result: Int = 1

        for _ in 0..<arity {
            guard let value = context.stack.popLast() else {
                throw OpError.stackIsEmpty
            }

            switch value {
            case let .dict(d):
                if list != nil {
                    throw OpError.typeMismatch(expected: [.int], actual: [value.type])
                }
                list = d

            case let .int(v):
                result *= v

            default:
                if list != nil {
                    throw OpError.typeMismatch(expected: [.int], actual: [value.type])
                } else {
                    throw OpError.typeMismatch(expected: [.int, .dict], actual: [value.type])
                }
            }
        }

        if let list = list {
            var product = HDict()
            for i in 0..<result {
                for (j, elem) in list.enumerated() {
                    product[.int(list.count * i + j)] = elem.value
                }
            }
            context.stack.append(.dict(product))

        } else {
            context.stack.append(.int(result))
        }
    }

    static func mod(context: inout Context) throws {
        guard let rhs = context.stack.popLast(),
              let lhs = context.stack.popLast()
        else {
            throw OpError.stackIsEmpty
        }

        let result: Value
        switch (lhs, rhs) {
        case let (.int(lhs), .int(rhs)):
            result = .int(lhs % rhs)
        default:
            throw OpError.typeMismatch(expected: [.int, .int], actual: [lhs.type, rhs.type])
        }
        context.stack.append(result)
    }

    static func notEquals(context: inout Context) throws {
        guard let rhs = context.stack.popLast(),
              let lhs = context.stack.popLast()
        else {
            throw OpError.stackIsEmpty
        }

        context.stack.append(.bool(lhs != rhs))
    }

    static func min(context: inout Context) throws {
        guard let value = context.stack.popLast() else {
            throw OpError.stackIsEmpty
        }

        switch value {
        case .set(let s):
            guard let m = s.min() else {
                throw OpError.setIsEmpty
            }

            context.stack.append(m)

        case .dict(let d):
            guard let m = d.values.min() else {
                throw OpError.dictIsEmpty
            }

            context.stack.append(m)

        default:
            throw OpError.typeMismatch(expected: [.set, .dict], actual: [value.type])
        }
    }

    static func max(context: inout Context) throws {
        guard let value = context.stack.popLast() else {
            throw OpError.stackIsEmpty
        }

        switch value {
        case .set(let s):
            guard let m = s.max() else {
                throw OpError.setIsEmpty
            }

            context.stack.append(m)

        case .dict(let d):
            guard let m = d.values.max() else {
                throw OpError.dictIsEmpty
            }

            context.stack.append(m)

        default:
            throw OpError.typeMismatch(expected: [.set, .dict], actual: [value.type])
        }
    }

    static func keys(context: inout Context) throws {
        guard case let .dict(value) = context.stack.popLast() else {
            throw OpError.stackTypeMismatch(expected: .dict)
        }

        context.stack.append(.set(HSet(value.keys.sorted())))
    }

}

extension Nary: CustomDebugStringConvertible {

    var debugDescription: String {
        switch self {
        case .minus:
            return "Nary.minus"
        case .negate:
            return "Nary.negate"
        case .not:
            return "Nary.not"
        case .equals:
            return "Nary.equals"
        case .notEquals:
            return "Nary.notEquals"
        case .dictAdd:
            return "Nary.dictAdd"
        case .plus(arity: let arity):
            return "Nary.plus(arity: \(String(reflecting: arity)))"
        case .atLabel:
            return "Nary.atLabel"
        case .range:
            return "Nary.range"
        case .isEmpty:
            return "Nary.isEmpty"
        case .len:
            return "Nary.len"
        case .setAdd:
            return "Nary.setAdd"
        case .keys:
            return "Nary.keys"
        case .lessThan:
            return "Nary.lessThan"
        case .greaterThan:
            return "Nary.greaterThan"
        case .lessThanOrEqual:
            return "Nary.lessThanOrEqual"
        case .greaterThanOrEqual:
            return "Nary.greaterThanOrEqual"
        case .in:
            return "Nary.in"
        case .min:
            return "Nary.min"
        case .max:
            return "Nary.max"
        case .union(arity: let arity):
            return "Nary.union(arity: \(arity))"
        case .getContext:
            return "Nary.getContext"
        case .times(arity: let arity):
            return "Nary.times(arity: \(arity))"
        case .mod:
            return "Nary.mod"
        }
    }

}
