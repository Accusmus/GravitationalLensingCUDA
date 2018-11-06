#include "fitsfile.h"

FitsFile::FitsFile() {
}

FitsFile::FitsFile(std::string filename) {
  open(filename, READONLY);
}

void FitsFile::open(std::string filename) {
  open(filename, READONLY);
}

void FitsFile::open(std::string filename, int mode) {
  int status = 0;
  if (fits_open_file(&fptr, filename.c_str(), mode, &status))
    handle_error(status);
}

void FitsFile::create_file(std::string filename)
{
  int status = 0;
  if (fits_create_file(&fptr, ("!" + filename).c_str(), &status))
    return handle_error(status);

}

void FitsFile::moveTo(int hduNum) {
}

int FitsFile::getAxisDim() {

  int status = 0;
  int naxis;
  char comment[FLEN_COMMENT];
  if (fits_read_key(fptr, TINT, "NAXIS", &naxis, comment, &status))
    handle_error(status);
  return naxis;

}

int FitsFile::getAxisSize(int axis_no) {

  int status = 0;
  char comment[FLEN_COMMENT];
  std::string keyname = "NAXIS" + to_string(axis_no);
  int size;
  if (fits_read_key(fptr, TINT, (char*)keyname.c_str(), &size, comment, &status))
    handle_error(status);
  return size;

}

void FitsFile::write_comment(std::string comment)
{
  int status = 0;
  if (fits_write_comment(fptr, (char*)comment.c_str(), &status))
    handle_error(status);
}

void FitsFile::write_history(std::string history)
{
  int status = 0;
  if (fits_write_history(fptr, (char*)history.c_str(), &status))
    handle_error(status);
}

void FitsFile::close() {
  int status = 0;
  if (fits_close_file(fptr, &status))
    handle_error(status);
}

FitsFile::~FitsFile() {
}

// TODO: replace with exception handler
void FitsFile::handle_error(int status)
{
  char emesg[FLEN_STATUS];
  fits_get_errstatus(status, emesg);

  std::string message = "FITSIO ERROR! " + std::string(emesg);
  throw FitsIOException(message);
}

FitsIOException::FitsIOException(std::string msg)
{
  message = msg;
}

FitsIOException::~FitsIOException() throw()
{
}
