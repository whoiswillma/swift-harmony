//
//  Nary.swift
//  
//
//  Created by William Ma on 10/29/21.
//

import Foundation

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
        guard let rhs = context.stack.popLast(),
              let lhs = context.stack.popLast()
        else {
            throw OpError.stackIsEmpty
        }

        context.stack.append(.bool(lhs == rhs))
    }

    static func range(context: inout Context) throws {
        guard case let .int(rhs) = context.stack.popLast(),
              case let .int(lhs) = context.stack.popLast()
        else {
            throw OpError.stackTypeMismatch(expected: .int)
        }

        let set = Set((lhs...rhs).map { .int($0) })
        context.stack.append(.set(set))
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
        } else if let index = set.elements.firstIndex(where: { value < $0 }) {
            set.insert(value, at: index)
        } else {
            set.append(value)
        }

        context.stack.append(.set(set))
    }

    static func times(context: inout Context, arity: Int) throws {
        assert(arity >= 1)

        var list: Dict?
        var result: Int = 1

        for i in 0..<arity {
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
            var product = Dict()
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

}
