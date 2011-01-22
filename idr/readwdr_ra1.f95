! Read 184-Byte WDR files: Seasat, Geosat, GFO and ERS-1/2
!
! * convert MJD to time in seconds since 1-Jan-1985 (aka utc85 or ESA time)
! * filter out points with undefined elevation values
! * check flags for 'retracked' and 'problem retracking'
! * save to ASCII or Binary format (same_original_name.txt or .bin)
! 
! To compile: gfortran -fconvert=big-endian thisprog.f95 -o thisprog 
! To run: ./thisprog file1 file2 file3 ... 
! For binary output set: ASCII = .false.
!
! Notes:
! Undefined values are set to -9999
!
! For WDR format see:
! http://icesat4.gsfc.nasa.gov/data_products/level1.html
!
! For examples on how to use the code see:
! www.fspaolo.net/code
!
! Fernando Paolo <fpaolo@ucsd.edu>
! October 5, 2010

module readwdr_ra1
  implicit none
  type WDRv4             ! 184-Byte WDR structure for version 4 
    character(2) id
    integer(2) retstat1
    integer(4) time
    integer(4) lat
    integer(4) lon
    integer(4) surf     
    integer(4) surfstat  ! same as IDR

    integer(2) noise     ! parameters describing the function used 
    integer(2) ampinc1   ! to fit the waveform for GSFC retracking
    integer(2) midpoint1
    integer(2) risetime1
    integer(2) ampinc2
    integer(2) midpoint2
    integer(2) risetime2
    integer(2) expdecay2 
    integer(2) slope12
    integer(2) wfmpeak
    integer(2) trackgate 
    integer(2) agc 
    integer(2) h13 
    integer(2) waveform(64)
    integer(2) sig0 
    integer(2) retstat2 
    integer(2) empty1
  endtype  
end module

program main
  use readwdr_ra1
  implicit none

  type(WDRv4) :: wdr

  character(2) :: id
  real(8) :: secrev, fsecrev, secdat, mjd, fday, utc85
  real(8) :: lat, lon, surf, agc, sig0 
  integer(4) :: orbit, surfstat, surfcheck
  integer(2) :: retstat
  integer(2) :: fret, fprob, fmode

  character(100) :: infile, outfile
  integer :: nfiles, npts, nvalidpts, nrec, iargc, narg, ios, i
  logical :: filecreated

  logical, parameter :: ASCII = .true.   ! .false. for Binary output   [edit]
  integer, parameter :: RECLEN = 184     ! record length in bytes
  real(8), parameter :: MJD85 = 46066.   ! 1-Jan-1985 00:00:00h in MJD
  real(8), parameter :: DAYSECS = 86400. ! 1 day in seconds 

  narg = iargc()
  if (narg < 1) stop 'usage: ./thisprog <wdrfiles> (see code for edition)'

  print *, 'processing files:', narg, '...'
 
  ! iterate over input files
  nfiles = 0
  npts = 0
  nvalidpts = 0
  do i = 1, narg
    call getarg(i, infile)     ! get arg in position i
    open(1, file=infile, status='old', access='direct', &
         form='unformatted', recl=RECLEN)
 
    ! iterate over records
    filecreated = .false.
    nrec = 1
    ios = 0
    do 
      read(1, rec=nrec, iostat=ios) wdr
      if (ios /= 0) exit        ! if EOF
      nrec = nrec + 1
      id = wdr%id
      
      ! WDR Rev Record
      if (id == 'WR') then  
        !-------------------------------------------------------
        orbit = wdr%time             ! orbit number
        mjd = wdr%lat                ! days (integer part) 
        secrev = wdr%lon             ! secs (integer part)
        fsecrev = wdr%surf / 1e6     ! secs (fractional part)
        !-------------------------------------------------------

      ! WDR data Record
      else if (id == 'WD') then
        !-------------------------------------------------------
        secdat = wdr%time / 1e6      ! secs (since time in Rev)
        lat = wdr%lat / 1e6          ! latitude (deg)
        lon = wdr%lon / 1e6          ! longitude (deg)
        surf = wdr%surf / 1e2        ! surface elevation (m)
        agc = wdr%agc / 1e2          ! automatic gain control (dB)
        sig0 = wdr%sig0 / 1e2        ! sigma naught (dB)

        retstat = wdr%retstat1       ! retracking status flags (15-0)
        surfstat = wdr%surfstat      ! surface status flags (31-0)
        surfcheck = wdr%surf         ! check whether surf is valid 
        !-------------------------------------------------------

        ! Flags: big-end: 0-15 (lsb) -> little-end: 15-0 (msb)
        !-------------------------------------------------------
        fprob = 0
        if (btest(retstat, 13) .or. &      ! wvfm spec shaped: 0=no, 1=yes
            btest(retstat, 11) .or. &      ! wvfm spec retracked: 0=no, 1=yes
            btest(retstat, 8)  .or. &      ! problem w/leading edge: 0=no, 1=yes
            btest(retstat, 7)) fprob = 1   ! problem retracking: 0=no, 1=yes
        fret = 0                              
        if (btest(retstat, 5)) fret = 1    ! wvfm retracked: 0=no, 1=yes
        fmode = 0                    
        if (btest(retstat, 0)) fmode = 1   ! mode: 0=ocean, 1=ice
        !-------------------------------------------------------

        npts = npts + 1
 
        !!! computations

        ! select pts
        if (surfcheck /= -9999) then
            
          nvalidpts = nvalidpts + 1

          ! fday: is "fraction" of a day
          ! mjd: is modified julian "days"
          ! utc85: is seconds passed since 1-Jan-1985 00:00:00h
          fday = (secrev + fsecrev + secdat) / DAYSECS  ! days
          utc85 = (mjd + fday - MJD85) * DAYSECS        ! secs

          !!! output

          if (ASCII .and. .not. filecreated) then 
            outfile = trim(infile) // '.txt'  ! ext for ASCII
            open(2, file=outfile, status='unknown', form='formatted')
            filecreated = .true.
            nfiles = nfiles + 1
          else if (.not. filecreated) then
            outfile = trim(infile) // '.bin'  ! ext for Binary
            open(2, file=outfile, status='unknown', form='unformatted')
            filecreated = .true.
            nfiles = nfiles + 1
          endif
 
          if (lon < 0) lon = lon + 360. ! output lon: 0/360

          if (ASCII) then
            write(2, '(i6, f20.6, 2f14.6, 3f14.3, 3i2)') & 
              orbit, utc85, lat, lon, surf, agc, sig0, fmode, fret, fprob
          else
            write(2) & 
              orbit, utc85, lat, lon, surf, agc, sig0, fmode, fret, fprob
          endif

        endif
      endif
    enddo
    if (filecreated) close(2)
    close(1)
  enddo

  !print *, 'number of records:     ', nrec-1
  print *, 'number of points:      ', npts 
  print *, 'number of valid points:', nvalidpts 
  print *, 'files created:         ', nfiles
  if (ASCII) then 
    print *, 'output extension: .txt' 
  else
    print *, 'output extension: .bin' 
  endif  

end program
