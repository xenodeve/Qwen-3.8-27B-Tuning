# The one place in scripts/ that asks the driver about a GPU.
#
# WHY THIS FILE EXISTS (2026-08-26).
#
# A second card was connected, and this is the language where that broke
# SILENTLY. Every script here used the same idiom:
#
#     $vram = (nvidia-smi --query-gpu=memory.used,memory.free `
#              --format=csv,noheader,nounits) -split '\s*,\s*'
#     $vram[0]   # used
#     $vram[1]   # free
#
# nvidia-smi answers per card, so with two installed the native command returns
# TWO lines, `-split` flattens them into a FOUR-element array, and $vram[0] and
# $vram[1] become the FIRST card's numbers. Measured that day: used=1481
# free=10517 -- the retired RTX 4070 SUPER, not the card serving the model.
#
# No error. No warning. Show-ServerStatus.ps1 would have reported the model
# resident based on a card that had nothing loaded on it.
#
# The Python half of the same defect raised ValueError and stopped the sweep.
# CLAUDE.md's north star is that the loud failure is the good one -- so the fix
# is not a better `-split`, it is that a reading names its card or refuses.
#
# Pinned by bench/tests/test_no_module_queries_every_gpu.py (issue #50).

# The RTX 5060 Ti 16 GB (sm_120). A UUID rather than an index because index is a
# position in an enumeration the BIOS and driver can reorder without anyone
# editing this file -- and after such a reorder an index keeps working while
# meaning a different card, which is the failure above with extra steps.
$script:ServedGpuUuid = 'GPU-059b90e2-2b5c-00b8-f3ba-f6dea8de083e'
$script:ServedGpuName = 'NVIDIA GeForce RTX 5060 Ti'

function Get-InstalledGpu {
    <#
    .SYNOPSIS
    Every card the driver can see, as objects. Never throws -- error messages
    call it to say what WAS found.
    #>
    $lines = & nvidia-smi --query-gpu=uuid,name --format=csv,noheader 2>$null
    foreach ($line in @($lines)) {
        if (-not $line) { continue }
        $parts = $line -split ',', 2
        if ($parts.Count -eq 2) {
            [pscustomobject]@{ Uuid = $parts[0].Trim(); Name = $parts[1].Trim() }
        }
    }
}

function Test-ServedGpuPresent {
    param([string]$Uuid = $script:ServedGpuUuid)
    return [bool](@(Get-InstalledGpu | Where-Object { $_.Uuid -eq $Uuid }).Count)
}

function Get-GpuVram {
    <#
    .SYNOPSIS
    Used and free MiB for ONE named card.

    .DESCRIPTION
    `-i <uuid>` is what makes the answer single-valued. Returns $null when the
    card is absent, and writes a warning saying so -- callers here are status
    displays and sweep loops that should say "unknown" rather than print a
    number belonging to different silicon.
    #>
    param([string]$Uuid = $script:ServedGpuUuid)

    $line = & nvidia-smi -i $Uuid --query-gpu=memory.used,memory.free `
                --format=csv,noheader,nounits 2>$null
    if ($LASTEXITCODE -ne 0 -or -not $line) {
        $found = (Get-InstalledGpu | ForEach-Object { $_.Name }) -join ', '
        Write-Warning "GPU $Uuid is not installed. Found: $found"
        return $null
    }
    # @($line)[0] and not $line[0]: with one card the native command returns a
    # STRING, and $line[0] would be its first CHARACTER. That is the same class
    # of bug as the one this file replaces, so it is spelled out rather than
    # left to the reader.
    $parts = @($line)[0] -split '\s*,\s*'
    if ($parts.Count -ne 2) {
        Write-Warning "unreadable VRAM line for ${Uuid}: $line"
        return $null
    }
    return [pscustomobject]@{
        Uuid = $Uuid
        Used = [int]$parts[0]
        Free = [int]$parts[1]
    }
}

function Get-GpuLink {
    <#
    .SYNOPSIS
    Current PCIe generation and width for one named card.

    .DESCRIPTION
    Both downtrain while the card is idle, so a reading taken between runs
    describes the driver's power state and not the slot. Sample it under load
    or do not quote it (issue #51, stage 4).
    #>
    param([string]$Uuid = $script:ServedGpuUuid)

    $line = & nvidia-smi -i $Uuid `
                --query-gpu=pcie.link.gen.current,pcie.link.width.current `
                --format=csv,noheader,nounits 2>$null
    if ($LASTEXITCODE -ne 0 -or -not $line) { return $null }
    $parts = @($line)[0] -split '\s*,\s*'
    if ($parts.Count -ne 2) { return $null }
    return [pscustomobject]@{ Gen = [int]$parts[0]; Width = [int]$parts[1] }
}
