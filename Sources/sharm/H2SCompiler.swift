//
//  H2SCompiler.swift
//  
//
//  Created by William Ma on 11/2/21.
//

import Foundation

class H2SCompiler {

    let code: [Op]

    init(code: [Op]) {
        self.code = code
    }

    func run() throws {
        for op in code {
            print("\(String(reflecting: op)),")
        }
    }

}
