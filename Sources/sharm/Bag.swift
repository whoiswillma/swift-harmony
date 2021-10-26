//
//  File.swift
//  
//
//  Created by William Ma on 10/26/21.
//

import Foundation

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
