//
//  File.swift
//  
//
//  Created by William Ma on 10/26/21.
//

import Foundation

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
