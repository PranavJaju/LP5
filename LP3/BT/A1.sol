//SPDX-License-Identifier: Unlicensed

pragma solidity ^0.4.25;

contract Bank
{
   int bal;
   constructor()public {
        bal=1;
   }
   function getBalance() view public returns (int){
    return bal;
   }
   function withdraw(int amt)  public returns (string memory) {
        if(amt<=bal){
          bal = bal - amt;
          return "OK";
        }else{
          return "NOT OK";
        }
   }
   function deposit(int amt) public{
        bal = bal + amt;
   }   
}
