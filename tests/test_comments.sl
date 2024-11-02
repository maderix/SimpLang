// This is a single-line comment
/* This is a 
   multi-line comment
*/

fn kernel_main() {
    // Initialize variables
    var x = 10.0;  // First number
    var y = 5.0;   // Second number

    /* Calculate various operations
       between x and y */
    var sum = x + y;    // Addition
    var diff = x - y;   // Subtraction
    var prod = x * y;   // Multiplication
    var quot = x / y;   // Division

    // Return the sum of all operations
    return sum + diff + prod + quot;
}
