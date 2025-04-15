//SPDX-License-Identifier: Unlicensed

pragma solidity ^0.8.0;

contract StudentData {
    struct Student {
        string name;
        uint age;
        string major;
    }

    Student[] internal students;  // An array to store student data

    // Add a new student to the array
    function addStudent(string memory _name, uint _age, string memory _major) public {
        Student memory newStudent = Student(_name, _age, _major);
        students.push(newStudent);
    }

    // Get the total number of students
    function getStudentCount() public view returns (uint) {
        return students.length;
    }

    // Get student data by index
    function getStudent(uint index) public view returns (string memory, uint, string memory) {
        require(index < students.length, "Student does not exist");
        Student memory student = students[index];
        return (student.name, student.age, student.major);
    }

    // Fallback function (not recommended for real-world use)
    receive() external payable {
        // This function can receive Ether, but it doesn't do anything in this example.
    }
}
