<#
    The status report, in ONE file, because two processes need it.

    `serve.ps1` prints it when a server is already running. The watcher it
    starts prints it when the server it is watching comes up. They are separate
    processes -- the watcher cannot call a function defined in the launcher --
    so without this the block would exist twice and drift, which is the
    duplication the launcher was simplified to remove.

    Everything here is READ, not assumed. Either caller may be looking at a
    server it did not start, so the bind comes from the listening socket rather
    than from any flag passed to anyone.
#>

function Show-ServerStatus {
    param($Props, [int]$Port = 8080, [int]$OnGpu = 0, [int]$Total = 0)

    $base = "http://127.0.0.1:$Port"
    $listen = @(Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue |
                ForEach-Object { $_.LocalAddress } | Sort-Object -Unique)
    $wide = $listen -contains '0.0.0.0' -or $listen -contains '::'

    Write-Host ""
    Write-Host "Serving on $base" -ForegroundColor Green
    Write-Host ("  model     {0}  ({1})" -f $Props.model_alias, $Props.model_ftype)
    Write-Host ("  build     {0}" -f $Props.build_info)
    $nctx = $Props.default_generation_settings.n_ctx
    if ($nctx) { Write-Host ("  window    {0:N0} tokens" -f $nctx) }

    # --fit SPILLS rather than refusing, and that reads as success in every
    # field except the layer count. Absent rather than guessed when unknown.
    if ($Total -gt 0) {
        if ($OnGpu -eq $Total) {
            Write-Host ("  layers    {0}/{1} on the GPU" -f $OnGpu, $Total) -ForegroundColor Green
        } else {
            Write-Host ("  layers    {0}/{1} -- SPILLED" -f $OnGpu, $Total) -ForegroundColor Red
        }
    } else {
        Write-Host "  layers    not seen in this stream" -ForegroundColor DarkGray
    }

    $vram = & nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader 2>$null
    if ($vram) { Write-Host ("  VRAM      {0}" -f ($vram -join '')) }

    try {
        $slots = Invoke-RestMethod -Uri "$base/slots" -TimeoutSec 4
        $used = @($slots | ForEach-Object { $_.n_prompt_tokens }) | Where-Object { $_ }
        if ($used) { Write-Host ("  context   {0:N0} tokens in the live slot" -f ($used | Measure-Object -Maximum).Maximum) }
    } catch { }

    Write-Host ("  bind      {0}" -f ($listen -join ', ')) -ForegroundColor $(if ($wide) { 'Yellow' } else { 'Green' })
    if ($wide) {
        Write-Host "  reachable from another machine at:" -ForegroundColor Yellow
        Get-NetIPAddress -AddressFamily IPv4 -ErrorAction SilentlyContinue |
            Where-Object { $_.IPAddress -ne '127.0.0.1' } |
            Sort-Object InterfaceAlias |
            ForEach-Object { Write-Host ("    http://{0}:{1}   ({2})" -f $_.IPAddress, $Port, $_.InterfaceAlias) }
    }
}
