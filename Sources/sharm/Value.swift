//
//  File.swift
//  
//
//  Created by William Ma on 10/26/21.
//

import Collections

typealias Dict = OrderedDictionary<Value, Value>

extension Dict: Comparable {

    public static func < (lhs: OrderedDictionary<Key, Value>, rhs: OrderedDictionary<Key, Value>) -> Bool {
        lhs.lexicographicallyPrecedes(rhs) { lhs, rhs in
            if lhs.key < rhs.key {
                return true
            } else {
                return lhs.value < rhs.value
            }
        }
    }

}

typealias Set = OrderedSet<Value>

extension Set: Comparable {

    public static func < (lhs: OrderedSet<Element>, rhs: OrderedSet<Element>) -> Bool {
        lhs.elements.lexicographicallyPrecedes(rhs.elements)
    }

}

extension Bool: Comparable {

    public static func < (lhs: Bool, rhs: Bool) -> Bool {
        lhs == false && rhs == true
    }

}

extension Array: Comparable where Element: Comparable {

    public static func < (lhs: Array<Element>, rhs: Array<Element>) -> Bool {
        lhs.lexicographicallyPrecedes(rhs)
    }

}

struct Context: Hashable {

    let name: String
    let entry: Int
    let arg: Value

    var stack: [Value]
    var pc: Int
    var fp: Int // unused?
    var vars: Dict // scoped-storage
    var atomicLevel: Int
    var readonlyLevel: Int

    var terminated: Bool = false
    var isAtomic: Bool {
        atomicLevel > 0
    }
    var isReadonly: Bool {
        readonlyLevel > 0
    }

    init(name: String, entry: Int, arg: Value, stack: [Value]) {
        self.name = name
        self.entry = entry
        self.arg = arg

        self.stack = stack
        self.pc = entry
        self.fp = 0
        self.vars = Dict()
        self.atomicLevel = 0
        self.readonlyLevel = 0
    }

}

extension Context: CustomStringConvertible {

    var description: String {
        return "Context(pc=\(pc),fp=\(fp),at=\(atomicLevel),rd=\(readonlyLevel),tm=\(terminated)\n"
            + "\tvars=\(vars)\n"
            + "\tstack=\(stack)\n"
            + ")"
    }

}

extension Context: Comparable {

    static func < (lhs: Context, rhs: Context) -> Bool {
        lhs.name < rhs.name
        && lhs.entry < rhs.entry
        && lhs.arg < rhs.arg
        && lhs.stack < rhs.stack
        && lhs.pc < rhs.pc
        && lhs.fp < rhs.fp
        && lhs.vars < rhs.vars
        && lhs.atomicLevel < rhs.atomicLevel
        && lhs.readonlyLevel < rhs.readonlyLevel
        && lhs.terminated < rhs.terminated
    }

}

enum Calltype: Int {
    case process = 1
    case normal = 2
}

// It is an important condition that these are in the same order as in Charm!
enum ValueType: Int, Hashable {

    case atom
    case bool
    case int
    case dict
    case address
    case pc
    case set
    case context

}

indirect enum Value: Hashable {

    case atom(String)
    case bool(Bool)
    case int(Int)
    case dict(Dict)
    case address([Value])
    case pc(Int)
    case set(Set)
    case context(Context)

    var type: ValueType {
        switch self {
        case .atom: return .atom
        case .bool: return .bool
        case .int: return .int
        case .dict: return .dict
        case .address: return .address
        case .pc: return .pc
        case .set: return .set
        case .context: return .context
        }
    }

    static let noneValue = Value.address([])

}

extension Value: Comparable {

    static func == (lhs: Value, rhs: Value) -> Bool {
        switch (lhs, rhs) {
        case let (.bool(lhs), .bool(rhs)):
            return lhs == rhs
        case let (.int(lhs), .int(rhs)):
            return lhs == rhs
        case let (.atom(lhs), .atom(rhs)):
            return lhs == rhs
        case let (.pc(lhs), .pc(rhs)):
            return lhs == rhs
        case let (.dict(lhs), .dict(rhs)):
            return lhs == rhs
        case let (.set(lhs), .set(rhs)):
            return lhs == rhs
        case let (.address(lhs), .address(rhs)):
            return lhs == rhs
        case let (.context(lhs), .context(rhs)):
            return lhs == rhs
        default:
            return false
        }
    }

    static func < (lhs: Value, rhs: Value) -> Bool {
        if lhs.type.rawValue < rhs.type.rawValue {
            return true
        }

        switch (lhs, rhs) {
        case let (.bool(lhs), .bool(rhs)):
            // lhs < rhs iff lhs = false and rhs = true
            return lhs < rhs
        case let (.int(lhs), .int(rhs)):
            return lhs < rhs
        case let (.atom(lhs), .atom(rhs)):
            return lhs < rhs
        case let (.pc(lhs), .pc(rhs)):
            return lhs < rhs
        case let (.dict(lhs), .dict(rhs)):
            return lhs < rhs
        case let (.set(lhs), .set(rhs)):
            return lhs < rhs
        case let (.address(lhs), .address(rhs)):
            return lhs < rhs
        case let (.context(lhs), .context(rhs)):
            return lhs < rhs
        default:
            fatalError()
        }
    }

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
            let value = try values.decode([Value].self, forKey: .value)
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
        case .atom(let value): return ".\(value)"
        case .bool(let value): return value ? "True" : "False"
        case .int(let value): return "\(value)"
        case .dict(let value): return "{\(value.map { "\($0):\($1)" }.joined(separator: ", "))}"
        case .noneValue: return "None"
        case .address(let value): return "?\(value.map { "\($0)" }.joined(separator: "."))"
        case .pc(let value): return "PC(\(value))"
        case .set(let value): return "{\(value.map { "\($0)" }.joined(separator: ", "))}"
        case .context(let context): return "\(context)"
        }
    }

}
