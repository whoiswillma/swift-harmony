//
//  File.swift
//  
//
//  Created by William Ma on 10/26/21.
//

import Foundation

struct Bag<T: Hashable> {

    struct Index {
        fileprivate let index: Dictionary<T, Int>.Index
    }

    private var collection: [T: Int]
    private(set) var count: Int = 0

    init() {
        collection = [:]
    }

    init(_ arr: [T]) {
        collection = [:]

        for elem in arr {
            add(elem)
        }
    }

    mutating func add(_ value: T) -> Index {
        collection[value, default: 0] += 1
        count += 1
        return index(for: value)!
    }

    func contains(_ value: T) -> Bool {
        collection[value] != nil
    }

    func count(of value: T) -> Int {
        collection[value, default: 0]
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

    func index(for value: T) -> Index? {
        collection.index(forKey: value).map { Index(index: $0) }
    }

    func get(index: Index) -> T {
        collection[index.index].key
    }

    mutating func remove(index: Index) -> T {
        let value = get(index: index)
        remove(value)
        return value
    }

    func elements() -> Swift.Set<T> {
        Swift.Set(collection.keys)
    }

    var startIndex: Index {
        Index(index: collection.startIndex)
    }

    var randomIndex: Index {
        index(for: collection.keys.randomElement()!)!
    }

}
