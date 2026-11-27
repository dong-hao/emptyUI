"""Read native Phoenix calibration through the manufacturer's SysCal V6."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

import numpy as np
from mt_metadata.timeseries.filters import FrequencyResponseTableFilter


def binary_chains(table, directory, ts_path, rate):
    executable = (os.environ.get("PHOENIX_SYSCAL") or shutil.which("SysCal_V6.exe")
                  or "")
    if not Path(executable).is_file():
        raise FileNotFoundError("Set PHOENIX_SYSCAL to the Phoenix SysCal_V6 executable")
    tags = table.tbl_dict
    level = ts_path.suffix.upper()[3:]
    if level not in ("2", "3", "4", "5"):
        raise ValueError(f"Unsupported SysCal recording level: {ts_path.suffix}")
    inputs = [directory / f"{int(tags['SNUM'])}.CLB"]
    for comp in table.channel_keys:
        if comp in ("hx", "hy", "hz"):
            inputs.append(directory / f"{tags[comp.upper()+'SN']}.CLC")
    # Linear mesh resolves instrument delay; extra low-frequency nodes cover long windows.
    frequencies = np.unique(np.r_[np.geomspace(1e-6,rate/512,64),
                                  rate*np.arange(1,205)/512])
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        for src in [table.file_path,*inputs]:
            shutil.copy2(src,folder/src.name)
        lines=[]
        for chunk in np.array_split(frequencies, (len(frequencies)+95)//96):
            (folder/"grid.pfc").write_text("FldType, 1,\n"+"".join(
                f"Frequency, {level}, {f:.12g},\n" for f in chunk),encoding="ascii")
            output=folder/"response.cts"
            output.unlink(missing_ok=True)
            proc=subprocess.run([executable,"grid.pfc",table.file_path.name,"*.clc","*.clb","response.cts"],
                                cwd=folder,timeout=120,creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0),capture_output=True)
            block=output.read_text(encoding="ascii").splitlines() if output.exists() else []
            if len(block)!=len(chunk)+1:
                raise RuntimeError(f"Incomplete SysCal response for TS{level}, {chunk[0]} to {chunk[-1]} Hz (exit {proc.returncode})")
            lines.extend(block[1:] if lines else block)
    serial,_,count=map(int,lines[0].strip().rstrip(",").split(",")[1:])
    data=np.asarray([[float(x) for x in line.strip().rstrip(",").split(",")]
                     for line in lines[1:] if line.strip()])
    if (serial!=int(tags["SNUM"]) or data.shape!=(len(frequencies),2+2*count)
            or not np.allclose(data[:,0],frequencies,rtol=6e-5,atol=6e-11)
            or not np.all(data[:,1]==int(level))):
        raise ValueError((serial,count,level,data.shape,len(frequencies),[(a,b) for a,b in zip(data[:,0],frequencies) if not np.isclose(a,b,rtol=6e-5,atol=6e-11)][:5]))
    chains={}
    for comp,n in table.channel_keys.items():
        if comp not in ("ex","ey","hx","hy","hz"):
            continue
        electric=comp.startswith("e")
        scale=getattr(table,comp+"_calibration")*1e-6 if electric else table.magnetic_calibration*1e-9
        response=(data[:,2*n]+1j*data[:,2*n+1])*2**23*scale
        if not np.isfinite(response).all() or np.any(abs(response)==0):
            raise ValueError(f"Invalid SysCal response for {comp}")
        unit="milliVolt per kilometer" if electric else "nanoTesla"
        phase=np.unwrap(np.angle(response))
        parts=max(1,int(np.ceil(np.max(abs(phase))/(5*np.pi))))
        stages=[]
        for i in range(parts):
            stage=FrequencyResponseTableFilter(frequencies=data[:,0],amplitudes=abs(response) if i==0 else np.ones(len(response)),
                phases=phase/parts,units_in=unit,units_out=unit,gain=1.)
            stage.name=f"syscal_{table.file_path.stem.lower()}_ts{level}_{comp}_{i}"
            stages.append(stage)
        recovered=np.prod([s.complex_response(data[:,0]) for s in stages],axis=0)
        if not np.allclose(recovered,response,rtol=1e-8,atol=0):
            raise ValueError(f"Phase encoding failed for {comp}")
        chains[comp]=stages
    return chains
