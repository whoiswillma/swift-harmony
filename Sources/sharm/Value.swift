//
//  Value.swift
//  
//
//  Created by William Ma on 10/26/21.
//

typealias HDict = OrderedDictionary<Value, Value>

extension HDict: Comparable {

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

typealias HSet = OrderedSet<Value>

extension HSet: Comparable {

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

struct Context {

    let name: String
    let entry: Int
    let arg: Value

    var stack: [Value]
    var pc: Int
    var vars: HDict // scoped-storage
    var atomicLevel: Int = 0
    var readonlyLevel: Int = 0

    var terminated: Bool = false

    var isAtomic: Bool = false // not the same as atomicLevel > 0 since there is lazy atomic
    var atomicPc: Int = -1 // the pc of an atomicInc in which this context became atomic

    var isReadonly: Bool {
        readonlyLevel > 0
    }

    init(name: String, entry: Int, arg: Value, stack: [Value]) {
        self.name = name
        self.entry = entry
        self.arg = arg
        self.stack = stack
        self.pc = entry

        self.vars = HDict()
    }

}

extension Context: Hashable {

    func hash(into hasher: inout Hasher) {
        name.hash(into: &hasher)
        entry.hash(into: &hasher)
        arg.hash(into: &hasher)
        pc.hash(into: &hasher)
        atomicLevel.hash(into: &hasher)
        readonlyLevel.hash(into: &hasher)
        terminated.hash(into: &hasher)
        isAtomic.hash(into: &hasher)
        atomicPc.hash(into: &hasher)
    }

}

extension Context: Comparable {

    static func < (lhs: Context, rhs: Context) -> Bool {
        lhs.name < rhs.name
        && lhs.entry < rhs.entry
        && lhs.arg < rhs.arg
        && lhs.stack < rhs.stack
        && lhs.pc < rhs.pc
        && lhs.vars < rhs.vars
        && lhs.atomicLevel < rhs.atomicLevel
        && lhs.readonlyLevel < rhs.readonlyLevel
        && lhs.terminated < rhs.terminated
    }

}

extension Context {

    static let initContext: Context = {
        var initContext = Context(name: "__init__", entry: 0, arg: .dict([:]), stack: [.dict([:])])
        initContext.atomicLevel = 1
        return initContext
    }()

}

extension Context: CustomStringConvertible {

    var description: String {
        return "\(name), [\(stack.map { $0.description }.joined(separator: ", "))]"
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
    case dict(HDict)
    case address([Value])
    case pc(Int)
    case set(HSet)
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
            var dict = HDict()
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
            let value = HSet(try values.decode([Value].self, forKey: .value))
            self = .set(value)

        default:
            fatalError()
        }
    }

}

extension Value: Encodable {

    func encode(to encoder: Encoder) throws {
        var values = encoder.container(keyedBy: CodingKeys.self)

        switch self {
        case .atom(let value):
            try values.encode("atom", forKey: .type)
            try values.encode(value, forKey: .value)

        case .bool(let value):
            try values.encode("bool", forKey: .type)
            try values.encode(value ? "True" : "False", forKey: .value)

        case .int(let value):
            try values.encode("int", forKey: .type)
            try values.encode(value.description, forKey: .value)

        case .dict(let dict):
            try values.encode("dict", forKey: .type)
            let dictEntries: [[String: Value]] = dict.map { key, value in
                ["key": key, "value": value]
            }
            try values.encode(dictEntries, forKey: .value)

        case .address(let indexPath):
            try values.encode("address", forKey: .type)
            try values.encode(indexPath, forKey: .value)

        case .pc(let pc):
            try values.encode("pc", forKey: .type)
            try values.encode(pc, forKey: .value)

        case .set(let set):
            try values.encode("set", forKey: .type)
            try values.encode(set.elements, forKey: .value)

        case .context:
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
        case .context: return "Context"
        }
    }

}

extension Value: CustomDebugStringConvertible {

    var debugDescription: String {
        switch self {
        case .atom(let value): return "Value.atom(\(value.debugDescription))"
        case .bool(let value): return "Value.bool(\(value))"
        case .int(let value): return "Value.int(\(value))"
        case .dict(let value):
            if value.isEmpty {
                return "Value.dict([:])"
            } else {
                return "Value.dict([\(value.elements.map { "\($0.debugDescription): \($1.debugDescription)" }.joined(separator: ", "))])"
            }
        case .address(let indexPath): return "Value.address(\(indexPath))"
        case .pc(let value): return "Value.pc(\(value))"
        case .set(let value): return "Value.set(\(value.debugDescription))"
        case .context: fatalError()
        }
    }

}
