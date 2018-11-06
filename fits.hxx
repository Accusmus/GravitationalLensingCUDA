#ifndef FITS_HXX
#define FITS_HXX

#include "fitsfile.h"

template<typename T>
int datatype(void) {return 0;}
//template<> int datatype<char>(void) {return TBYTE;}
template<> int datatype<unsigned short>(void) {return TUSHORT;}
template<> int datatype<short>(void) {return TSHORT;}
template<> int datatype<int>(void) {return TINT;}
template<> int datatype<float>(void) {return TFLOAT;}
template<> int datatype<double>(void) {return TDOUBLE;}
template<> int datatype<std::string>(void) {return TSTRING;}

template<typename T>
int bitpix(void) {return 0;}

template<> int bitpix<char>(void) {return BYTE_IMG;}
template<> int bitpix<unsigned short>(void) {return USHORT_IMG;}
template<> int bitpix<short>(void) {return SHORT_IMG;}
template<> int bitpix<int>(void) {return LONG_IMG;}
template<> int bitpix<float>(void) {return FLOAT_IMG;}
template<> int bitpix<double>(void) {return DOUBLE_IMG;}

// Type safe convenience functions

template<typename T>
void FitsFile::read_data(T* pixdata, int num_pixels) {

  long firstelem = 1;
  int anynull;
  T nullvalue;
  int status = 0;

  int dtype = datatype<T>();

  if (fits_read_img(fptr, dtype, firstelem, num_pixels, 
		    &nullvalue, pixdata, &anynull, &status)) 
    handle_error(status);

}

template<typename BITPIX>
void FitsFile::create_img_2d(int npixx, int npixy)
{
  long naxes[2];
  naxes[0] = npixx;
  naxes[1] = npixy;
  int status = 0;
  int bpix = bitpix<BITPIX>();
  if (fits_create_img(fptr, bpix, 2, naxes, &status))
    handle_error(status);
  
}

template<typename T>
void FitsFile::write_image(T* pixdata, int num_pixels) {

  long fpixel[2];
  fpixel[0] = 1;
  fpixel[1] = 1;
  long npixels = num_pixels;
  int dtype = datatype<T>();
  int status = 0;
  if (fits_write_pix(fptr, dtype, fpixel, npixels, pixdata, &status))
    handle_error(status);

}

// Read FITS keyward value, discard comment
template<typename T>
T FitsFile::read_key(std::string keyname)
{
  char comment[FLEN_COMMENT];
  int status = 0;
  int dtype = datatype<T>();
  T value;
  if (fits_read_key(fptr, dtype, (char*)keyname.c_str(), &value, comment, &status))
    handle_error(status);
  return value;
}
// Specialist function to deal with C++ strings
template<>
std::string FitsFile::read_key<std::string>(std::string keyname)
{
  char comment[FLEN_COMMENT];
  int status = 0;
  int dtype = datatype<std::string>();
  char value[FLEN_VALUE];
  if (fits_read_key(fptr, dtype, (char*)keyname.c_str(), &value, comment, &status))
    handle_error(status);
  return std::string(value);
}

template<typename T>
void FitsFile::write_key(std::string keyname, T value, std::string comment)
{
  int status = 0;
  int dtype = datatype<T>();
  if (fits_write_key(fptr, dtype, (char*)keyname.c_str(), &value, (char*)comment.c_str(), &status))
    handle_error(status);
}
template<>
void FitsFile::write_key<std::string>(std::string keyname, std::string value, std::string comment)
{
  int status = 0;
  int dtype = datatype<std::string>();
  if (fits_write_key(fptr, dtype, (char*)keyname.c_str(), (char*)value.c_str(), (char*)comment.c_str(), &status))
    handle_error(status);
}
#endif


