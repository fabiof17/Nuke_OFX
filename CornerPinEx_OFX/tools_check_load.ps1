# Used by check_install.bat: loads the plugin DLL like Nuke does and reports errors.
param([string]$Path)
Add-Type -Namespace W -Name K -MemberDefinition @"
[DllImport("kernel32.dll", SetLastError=true, CharSet=CharSet.Unicode)] public static extern IntPtr LoadLibraryW(string f);
[DllImport("kernel32.dll", CharSet=CharSet.Ansi)] public static extern IntPtr GetProcAddress(IntPtr h, string n);
"@
$h = [W.K]::LoadLibraryW($Path)
if ($h -eq [IntPtr]::Zero) {
    $e = [Runtime.InteropServices.Marshal]::GetLastWin32Error()
    Write-Host "    FAILED: Windows cannot load the DLL (error $e)."
    if ($e -eq 126) {
        Write-Host "    A DLL the plugin depends on is missing: install the Visual C++ Redistributable x64"
        Write-Host "    https://aka.ms/vs/17/release/vc_redist.x64.exe"
    } elseif ($e -eq 193) {
        Write-Host "    The file is not a valid 64-bit DLL: rebuild with 'build.bat clean'."
    }
    exit 1
}
if ([W.K]::GetProcAddress($h, "OfxGetNumberOfPlugins") -eq [IntPtr]::Zero) {
    Write-Host "    FAILED: the DLL does not contain the OFX entry points."
    exit 1
}
Write-Host "    OK: the DLL loads and exposes the OFX entry points."
exit 0
