! Read 112-Byte IDR files: Envisat 
!
! * convert MJD to time in seconds since 1-Jan-1985 (aka utc85 or ESA time)
! * unapply tide correction (if applied), for later use of better ones
! * apply selected increment for orbit correction (if defined)
! * check flags for 'retracked' and 'problem retracking'
! * filter out points with undefined elevation values
! * filter out points with unavailable geophysical corrections
! * save to ASCII format (same_original_name.txt)
! 
! To compile: gfortran -fconvert=big-endian thisprog.f95 -o thisprog 
! To run: ./thisprog file1 file2 file3 ... 
! For binary output set: ASCII = .false.
!
! Notes:
! ------
! Undefined surf values are set to -9999
! Undefined corrections are set to 32767
! For Envisat no need to increment !!!
!
! For examples on how to use the code see:
! http://fspaolo.net/code
!
! For editing:
! check the '[edit]' keyword within the code
!
! For IDR format see:
! http://icesat4.gsfc.nasa.gov/data_products/level2.html
!
! Fernando Paolo <fpaolo@ucsd.edu>
! September 20, 2010

module readidr_ra2
  implicit none
  type IDRv4     ! 112-Byte IDR structure for version 4 Envisat
    character(2) id
    integer(2) retstat1
    integer(4) time
    integer(4) lat
    integer(4) lon
    integer(4) surf
    integer(4) wdr
    integer(4) altim
    integer(4) altstat
    integer(4) surfstat
    integer(2) iono
    integer(2) wet1
    integer(2) dry
    integer(2) geoid
    integer(2) stide 
    integer(2) otide 
    integer(2) slope 
    integer(2) swh
    integer(2) agc
    integer(2) att
    integer(2) empty1
    integer(2) inc1
    integer(2) empty2
    integer(2) inc2
    integer(2) empty3
    integer(2) inc3
    integer(2) retramp1
    integer(2) retramp2
    integer(2) sigramp1
    integer(2) sigramp2
    integer(2) tanslope
    integer(2) empty4
    integer(2) wet2

    integer(2) surfflag  ! ERS: modstat 
    integer(4) confflag  ! ERS: locstat and rshstat
    integer(2) instflag  ! ERS: wwlsstat
    integer(2) retflag   ! ERS: opstat

    integer(2) thres10 
    integer(2) thres20
    integer(2) thres50
    integer(2) retstat2

    integer(2) oretcor   ! 12B extra in Envisat
    integer(2) i1retcor
    integer(2) i2retcor
    integer(2) siretcor
    integer(2) instcor
    integer(2) doppcor
  endtype  
end module

program main
  use readidr_ra2
  implicit none

  type(IDRv4) :: idr

  character(2) :: id
  real(8) :: secrev, fsecrev, secdat, mjd, fday, utc85
  real(8) :: lat, lon, surf, otide, agc, inc 
  integer(4) :: orbit, surfstat, surfcheck
  integer(2) :: retstat, ionocheck, wetcheck, drycheck
  integer(2) :: stidecheck, otidecheck, slopecheck, inccheck
  integer(2) :: fret, fprob, fmode, fotide

  character(100) :: infile, outfile
  integer :: nfiles, npts, nvalidpts, nrec, iargc, narg, ios, i
  logical :: filecreated

  logical, parameter :: ASCII = .true.   ! .false. for Binary output   [edit]
  integer, parameter :: RECLEN = 112     ! record length in bytes: Envisat
  real(8), parameter :: MJD85 = 46066.   ! 1-Jan-1985 00:00:00h in MJD
  real(8), parameter :: DAYSECS = 86400. ! 1 day in seconds 

  narg = iargc()
  if (narg < 1) stop 'usage: ./thisprog <idrfiles> (see code for edition)'

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
      read(1, rec=nrec, iostat=ios) idr
      if (ios /= 0) exit        ! if EOF
      nrec = nrec + 1
      id = idr%id
      
      ! IDR Rev Record
      if (id == 'IR') then  
        !-------------------------------------------------------
        orbit = idr%time             ! orbit number
        mjd = idr%lat                ! days (integer part) 
        secrev = idr%lon             ! secs (integer part)
        fsecrev = idr%surf / 1e6     ! secs (fractional part)
        !-------------------------------------------------------

      ! IDR data Record
      else if (id == 'ID') then
        !-------------------------------------------------------
        secdat = idr%time / 1e6      ! secs (since time in Rev)
        lat = idr%lat / 1e6          ! latitude (deg)
        lon = idr%lon / 1e6          ! longitude (deg)
        surf = idr%surf / 1e2        ! surface elevation (m)
        otide = idr%otide / 1e3      ! ocean tide correction (m)
        agc = idr%agc / 1e2          ! automatic gain control (dB)
        inc = idr%inc3 / 1e2         ! orbit correction (m)            [edit]

        retstat = idr%retstat1       ! retracking status flags (15-0)
        surfstat = idr%surfstat      ! surface status flags (31-0)
        surfcheck = idr%surf         ! check whether surf is valid 
        ionocheck = idr%iono         ! check ionosphere
        wetcheck = idr%wet1          ! check wet tropo
        drycheck = idr%dry           ! check dry tropo
        stidecheck = idr%stide       ! check solid tide
        otidecheck = idr%otide       ! check ocean tide
        slopecheck = idr%slope       ! check slope corr
        inccheck = idr%inc3          ! check orbit increment           [edit]
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
        fmode = 3
        if (.not. btest(retstat, 1) .and. &     ! tracking mode (14 and 15):
            .not. btest(retstat, 0)) fmode = 0  ! 0 0 = Fine (~ocean in ERS)
        if (.not. btest(retstat, 1) .and. &
                  btest(retstat, 0)) fmode = 1  ! 0 1 = Medium (~ice in ERS)
        if (      btest(retstat, 1) .and. &
            .not. btest(retstat, 0)) fmode = 2  ! 1 0 = Coarse (no in ERS)
        fotide = 0
        if (btest(surfstat, 7)) fotide = 1 ! otide cor applied: 0=no, 1=yes
        !-------------------------------------------------------

        npts = npts + 1
 
        !!! computations

        ! select pts with available geophysical corr
        if ( &
            surfcheck /= -9999 .and. &
            ionocheck /= 32767 .and. &
            wetcheck /= 32767 .and. &
            drycheck /= 32767 .and. &
            stidecheck /= 32767 &
            !slopecheck /= 32767 &
            ) then
            
          nvalidpts = nvalidpts + 1

          ! fday: is "fraction" of a day
          ! mjd: is modified julian "days"
          ! utc85: is seconds passed since 1-Jan-1985 00:00:00h
          fday = (secrev + fsecrev + secdat) / DAYSECS  ! days
          utc85 = (mjd + fday - MJD85) * DAYSECS        ! secs

          ! add increment and detide
          !if (inccheck /= 32767) surf = surf + inc  ! no inc for Envisat
          if (fotide == 1) surf = surf + otide
 
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
 
          if (lon < 0) lon = lon + 360.       ! output lon: 0/360

          if (ASCII) then
            write(2, '(i6, f20.6, 2f14.6, 2f14.3, 3i2)') & 
              orbit, utc85, lat, lon, surf, agc, fmode, fret, fprob
          else
            write(2) & 
              orbit, utc85, lat, lon, surf, agc, fmode, fret, fprob
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
