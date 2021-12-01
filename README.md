# sharm

Sharm is an interpreter, model checker, and bytecode-to-source compiler, and bytecode-to-source model checker for 
Harmony. 

### Build

In order to use Sharm you must first install [Harmony](https://harmony.cs.cornell.edu/). 

After that, building Sharm itself is one command:

 1. `swift build -c release`
 
The compiled binary is located at `.build/release/sharm`, though the Sharm repository contains a symlink to it in the root directory of the project. 
 
### Tutorial

Sharm includes two demo programs, `PetersonCorrect.hny` and `PetersonWrong.hny`, which are altered implementations of  Peterson's algorithm from the Harmony book. PetersonCorrect includes log statements when a thread enters and exits the critical section, and PetersonWrong includes those log statements and modifies the critical section assertion to be wrong some of the time.

 1. **Run each program once under Harmony** 
 
    ```
    harmony PetersonCorrect.hny
    harmony PetersonWrong.hny
    ```
    
    Unsurprisingly, PetersonCorrect reports no issues and PetersonWrong reports a safety violation under Harmony.

 2. **Run the interpreting executor**
 
    ```
    ./sharm ie PetersonCorrect.hvm
    
    {0:1, 1:.enter}
    {0:1, 1:.exit}
    {0:0, 1:.enter}
    {0:0, 1:.exit}
    ```
    
    Try running this a few times. Based on whether a thread decides to contest the critical section, you may see lots of messages or none. In the execution above, thread one entered then exited the critical section, then thread zero entered and exited as well. You should never see two threads enter though! Sharm can print the non-deterministic choices it made during this execution with the `--print-history` flag. 
    
    ```
    ./sharm ie PetersonCorrect.hvm --print-history 
    ```
    
    > The only difference between PetersonWrong and PetersonCorrect is that the critical section assertion is made to only be correct some of the time
    
    ```
    diff PetersonCorrect.hny PetersonWrong.hny 
    15c15
    <         @cs: assert atLabel(cs) == { (thread, self): 1 }
    ---
    >         @cs: assert atLabel(cs) == { (thread, self): choose({ 0, 1 }) }
    ```
    
    > Based on this, what do you think will happen? Try running `./sharm ie PetersonWrong.hvm --print-history` a few times and observing when assertionFailures occur and what the last line in the history is.
    
 3. **Run the interpreting model checker**
  
    Let's run the interpreting model checker on PetersonCorrect:
    
    ```
    ./sharm imc PetersonCorrect.hvm 
    0 0
    1000 16
    No errors found
    Total states: 1604
    ```
    
    The first two lines print the number of visited and boundary states every thousand visited states. 
    
    > What do you think will happen if we model check PetersonWrong.hvm? How will this change if we execute it repeatedly?

 4. **Run the compiling executor**

    We can compile Harmony code into an executable
    
    ```
    ./sharm ce PetersonCorrect.hvm
    ```
    
    This will output Swift files into the directory `./sharmcompile/`. If we cd into the directory we can then build and run the executable with `./run.sh`.
    
 5. **Run the compiling model checker**
 
    Finally, let's compile the Harmony code into a Swift program that model checks the original code.
    
    ```
    ./sharm cmc PetersonCorrect.hvm
    ```
    
    Once again this will output Swift files into the directory `./sharmcompile/`, potentially overwriting ones that were already there. Running `./run.sh` again will build and run the model checker, which should find no errors.
