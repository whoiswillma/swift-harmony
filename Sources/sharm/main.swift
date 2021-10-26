#!/usr/bin/env swift

import Foundation
import OrderedCollections

struct Bag<T: Hashable> {

    struct Index {
        let index: Dictionary<T, Int>.Index
    }

    private var collection: [T: Int]
    private(set) var count: Int = 0

    mutating func add(_ value: T) {
        collection[value, default: 0] += 1
        count += 1
    }

    func contains(_ value: T) -> Bool {
        return collection[value] != nil
    }

    func count(of value: T) -> Int {
        return collection[value, default: 0]
    }

    mutating func remove(_ value: T) {
        if let elemCount = collection[value] {
            assert(elemCount > 0)
            let newElemCount = elemCount - 1
            if newElemCount > 0 {
                collection[value] = newElemCount
            } else {
                collection[value] = nil
            }
            count -= 1
        }
    }

    func index(forKey value: T) -> Index? {
        return collection.index(forKey: value).map { Index(index: $0) }
    }

    func get(index: Index) -> T {
        return collection[index.index].key
    }

    mutating func remove(index: Index) -> T {
        let value = get(index: index)
        remove(value)
        return value
    }

}

typealias Dict = OrderedDictionary<Value, Value>

indirect enum Value: Hashable {

    case noneVal
    case atom(String)
    case bool(Bool)
    case int(Int)
    case dict(Dict)
    case address(Value)
    case pc(Int)
    case set(Set<Value>)

}

extension Value: Decodable {

    enum CodingKeys: String, CodingKey {
        case type
        case value
    }

    init(from decoder: Decoder) throws {
        let values = try decoder.container(keyedBy: CodingKeys.self)
        let type = try values.decode(String.self, forKey: .type)

        switch type {
        case "atom":
            let value = try values.decode(String.self, forKey: .value)
            self = .atom(value)

        case "bool":
            let value = try values.decode(String.self, forKey: .value)
            assert(value == "True" || value == "False")
            self = .bool(value == "True")
        
        case "int":
            let value = Int(try values.decode(String.self, forKey: .value))!
            self = .int(value)

        case "dict":
            let dictEntries = try values.decode([[String: Value]].self, forKey: .value)
            var dict = Dict()
            for entry in dictEntries {
                let key = entry["key"]!
                let value = entry["value"]!
                dict[key] = value
            }
            self = .dict(dict)

        case "address":
            let value = try values.decode(Value.self, forKey: .value)
            self = .address(value)

        case "pc":
            let value = Int(try values.decode(String.self, forKey: .value))!
            self = .pc(value)

        case "set":
            let value = Set(try values.decode([Value].self, forKey: .value))
            self = .set(value)

        default:
            fatalError()
        }
    }

}

extension Value: CustomStringConvertible {

    var description: String {
        switch self {
        case .noneVal: return "None"
        case .atom(let value): return ".\(value)"
        case .bool(let value): return value ? "True" : "False"
        case .int(let value): return "\(value)"
        case .dict(let value): return "{\(value.map { "\($0):\($1)" }.joined(separator: ", "))}"
        case .address(let value): return "?\(value)"
        case .pc(let value): return "PC(\(value))"
        case .set(let value): return "{\(value.map { "\($0)" }.joined(separator: ", "))}"
        }
    }

}

struct Context: Hashable {

    var stack: [Value] = [.dict(Dict())]
    var pc: Int = 0
    var fp: Int = 0 // unused?
    var vars: Dict = Dict()
    var atomicLevel: Int = 0
    var readonlyLevel: Int = 0
    var terminated: Bool = false

}

extension Context: CustomStringConvertible {

    var description: String {
        return "Context(pc=\(pc),fp=\(fp),at=\(atomicLevel),rd=\(readonlyLevel),tm=\(terminated)\n"
            + "\tvars=\(vars)\n"
            + "\tstack=\(stack)\n"
            + ")"
    }

}

protocol Nondeterminism {

    mutating func choose(_ values: Bag<Value>) -> Bag<Value>.Index
    mutating func choose(_ context: Bag<Context>) -> Bag<Context>.Index

}

struct State {

    var nondeterminism: Nondeterminism
    var contexts: Bag<Context>
    var current: Bag<Context>.Index

}

enum Nary: String, Hashable {
    
    case minus
    case not
    case equals
    case dictAdd
    case plus

    var arity: Int {
        switch self {
        case .minus: return 2
        case .not: return 1
        case .equals: return 2
        case .dictAdd: return 3
        case .plus: return 2
        }
    }

}

enum Op: Hashable {

    case frame(name: String, params: [String])
    case push(Value)
    case sequential
    case choose
    case store(Value)
    case storeAddress
    case storeVar(String)
    case jump(pc: Int)
    case jumpCond(pc: Int, cond: Value)
    case loadVar(String)
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

        default:
            fatalError(op)
        }
    }

}

extension Op {

    func apply(_ context: inout Context) {
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

                switch (lhs, rhs) {
                case let (.int(lhs), .int(rhs)):
                    context.stack.append(.int(lhs + rhs))

                default:
                    fatalError()
                }

            default:
                fatalError()
            }

            context.pc += 1

        case let .storeVar(varName):
            let value = context.stack.popLast()!
            context.vars[.atom(varName)] = value
            context.pc += 1

        case .ret:
            let result = context.vars[.atom("result")] ?? .noneVal

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

        default:
            fatalError()
        }
    }

}

struct HVM: Decodable {
    let code: [Op]
}

let url = URL(fileURLWithPath: "/Users/williamma/Documents/sharm/multiargs.hvm")
let hvmData = try Data(contentsOf: url)
let hvm = try JSONDecoder().decode(HVM.self, from: hvmData)

var context = Context()
while !context.terminated {
    print(context)
    print(hvm.code[context.pc])
    hvm.code[context.pc].apply(&context)
}
print(context)
