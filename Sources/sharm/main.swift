#!/usr/bin/env swift

import Foundation
import OrderedCollections

struct HVM: Decodable {
    let code: [Op]
}

let url = URL(fileURLWithPath: "/Users/williamma/Documents/sharm/Diners.hvm")
let hvmData = try Data(contentsOf: url)
let hvm = try JSONDecoder().decode(HVM.self, from: hvmData)

let nondeterminism = BookkeepingNondeterminism()
defer {
    print("History")
    for elem in nondeterminism.history {
        switch elem {
        case .index(let i, let s): print("\tChose \(i) out of \(s)")
        case .context(let i, let s): print("\tChose \(i) out of \(s)")
        }
    }
}

private let interpreter = Interpreter(code: hvm.code, nondeterminism: nondeterminism)

while !interpreter.allTerminated {
    try interpreter.step()
}
