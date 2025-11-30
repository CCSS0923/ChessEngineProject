@echo off
setlocal

echo ============================
echo   CLEAN BUILD (FULL RESET)
echo ============================

:: 1) VS / CMake / Explorer / Google 드라이브 관련 프로세스 종료
echo [1] 프로세스 종료...
taskkill /IM devenv.exe /F >nul 2>&1
taskkill /IM cmake.exe /F >nul 2>&1
taskkill /IM ninja.exe /F >nul 2>&1
taskkill /IM explorer.exe /F >nul 2>&1
taskkill /IM GoogleDrive.exe /F >nul 2>&1
taskkill /IM GoogleDriveFS.exe /F >nul 2>&1

:: 2) 2초 대기 (파일 lock 해제)
timeout /T 2 >nul

:: 3) build 폴더 삭제
echo [2] build/ 삭제...
rmdir /S /Q build 2>nul

:: 4) 상위의 잔여 CMake 캐시 삭제
echo [3] CMake 캐시 삭제...
del /Q CMakeCache.txt 2>nul
rmdir /S /Q CMakeFiles 2>nul

:: 5) VSCode IntelliSense 캐시 삭제(.vs 폴더)
echo [4] .vs 폴더 삭제...
rmdir /S /Q .vs 2>nul

:: 6) 다시 explorer 실행
echo [5] Explorer 재시작...
start explorer.exe

echo ============================
echo        완료
echo ============================

endlocal
