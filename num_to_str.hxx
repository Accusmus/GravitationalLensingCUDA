#ifndef NUMTOSTR_H
#define NUMTOSTR_H

#include <sstream>

// Convenience function to use string streams to convert a numeric
// value into a string

template<typename T>
std::string to_string(T value) {

  std::ostringstream sout;
  sout << value;
  
  return sout.str();

}

#endif
