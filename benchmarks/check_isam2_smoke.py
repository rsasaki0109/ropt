"""Offline shared SE2 stream smoke; native iSAM2 is optional."""
from pathlib import Path
import subprocess
import sys
import tempfile

from paired_isam2 import agreement, validate
from pose_graph import require
from se2_stream import write


def main():
    require(len(sys.argv) in (2,3), 'usage: check_isam2_smoke.py ropt_binary [native_binary]')
    binaries=[Path(p).resolve() for p in sys.argv[1:]]
    with tempfile.TemporaryDirectory(prefix='ropt-isam2-smoke-') as tmp:
        root=Path(tmp); data=root/'input.stream'; write(data,25)
        checked=[]
        for index,binary in enumerate(binaries):
            output=root/f'result{index}'
            subprocess.run([str(binary),str(data),str(output),'1','0'],check=True)
            checked.append(validate(output,25))
            original={p.name:p.read_bytes() for p in output.iterdir()}
            failed=subprocess.run([str(binary),str(data),str(output),'1','0'],capture_output=True)
            require(failed.returncode!=0 and original=={p.name:p.read_bytes() for p in output.iterdir()}, 'existing output changed')
            for repeat in ('0','-1'):
                invalid=root/f'invalid-{index}-{repeat}'
                failed=subprocess.run([str(binary),str(data),str(invalid),repeat,'0'],capture_output=True)
                require(failed.returncode!=0 and not invalid.exists(),'invalid repetitions created output')
        if len(checked)==2:
            a,b=checked
            agreement(a[0],b[0],a[1],b[1])
    print('Shared nonlinear smoke passed: original-energy stationarity, full means, output/argument protection')


if __name__=='__main__': main()
