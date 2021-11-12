//
//  BasicBlock.swift
//  
//
//  Created by William Ma on 11/11/21.
//

import Foundation

struct BasicBlock {
    let pc: Int
    let ops: [Op]
    let endPc: Int
}
