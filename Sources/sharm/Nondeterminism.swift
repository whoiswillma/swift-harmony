//
//  Nondeterminism.swift
//  
//
//  Created by William Ma on 10/26/21.
//

import Foundation

protocol Nondeterminism {

    func chooseIndex(_ values: Set) -> Int
    func chooseContext(_ context: Bag<Context>) -> Bag<Context>.Index

}
