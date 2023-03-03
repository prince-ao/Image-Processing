#include<stdio.h>
#include<stdlib.h>
#include <unistd.h>

#define MAXBUF 5000

int main(void) {
  write(STDOUT_FILENO, "Enter one character: ", 21); // cout << "Enter one character: ";
  char aChar; // char buffer
  char buffer[MAXBUF]; // string buffer
  int byte = 0; // create index
  
  while(read(STDIN_FILENO, &aChar, 1) > 0){ // while while no error
    if(aChar == '\n') // if new line, break
      break;
    buffer[byte++] = aChar; // store in string buffer
  }
  buffer[byte] = 0; // end string with 0

  write(STDOUT_FILENO, "You entered: ", 13); // cout << "You entered: "

  byte = 0;
  while(buffer[byte] != 0) // cout << buffer;
    write(STDOUT_FILENO, &buffer[byte++], 1);
  write(STDOUT_FILENO, "\n", 1); // endl;

  return 0;
}