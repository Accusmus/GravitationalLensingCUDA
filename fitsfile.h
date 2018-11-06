// -*- c++ -*-
//
#ifndef FITSFILE_H
#define FITSFILE_H

#include <string>
#include <iostream>
#include <exception>

#include "fitsio.h"

#include "num_to_str.hxx"

class FitsFile {

 public:
  FitsFile();
  FitsFile(std::string filename);
  
  void open(std::string filename);
  void open(std::string filename, int mode);

  void create_file(std::string filename);

  int getAxisDim();
  int getAxisSize(int naxis);

  void moveTo(int hduNum);

  void write_comment(std::string comment);

  void write_history(std::string history);
  
  void close();
  
  ~FitsFile();

  template<typename T>
  void read_data(T* pixdata, int num_pixels);

  template<typename BITPIX>
  void create_img_2d(int npixx, int npixy);

  template<typename T>
  void write_image(T* pixdata, int num_pixels);

  template<typename T>
  T read_key(std::string keyname);

  template<typename T>
  void write_key(std::string keyname, T value, std::string comment);

  //private:
  fitsfile* fptr;

  static void handle_error(int status);

};

class FitsIOException : public std::exception {

public:
  FitsIOException(std::string msg);
  ~FitsIOException() throw();
  std::string message;
};


#endif
