"""Self-process Mach memory counters, with ABI checked against Apple SDK.

Source: local macOS SDK mach/task_info.h lines91,344-432, mach/arm/vm_types.h
and mach/mach_init.h. The prefix through phys_footprint is TASK_VM_INFO_REV1.
The rev1 ABI was checked against the Apple SDK with the C compiler.

All *_bytes fields are byte counters. `compressed` is the task's compressed
memory ledger, not physical compressor occupancy or a per-process swap meter.
`phys_footprint` is a task ledger accounting measure, not RSS. Do not add these
fields together as though they were disjoint physical allocations.
"""

import ctypes
import sys


class TaskVmInfoRev1(ctypes.Structure):
    _pack_ = 4
    _fields_ = [
        ("virtual_size", ctypes.c_uint64),
        ("region_count", ctypes.c_int32),
        ("page_size", ctypes.c_int32),
    ] + [
        (name, ctypes.c_uint64)
        for name in (
            "resident_size",
            "resident_size_peak",
            "device",
            "device_peak",
            "internal",
            "internal_peak",
            "external",
            "external_peak",
            "reusable",
            "reusable_peak",
            "purgeable_volatile_pmap",
            "purgeable_volatile_resident",
            "purgeable_volatile_virtual",
            "compressed",
            "compressed_peak",
            "compressed_lifetime",
            "phys_footprint",
        )
    ]


TASK_VM_INFO = 22
REV1_COUNT = ctypes.sizeof(TaskVmInfoRev1) // ctypes.sizeof(ctypes.c_uint32)
_lib = None


def sample_self():
    """Read the calling process only. Fail explicitly on ABI/API failure."""
    global _lib
    if sys.platform != "darwin":
        raise RuntimeError("Mach memory telemetry requires macOS")
    if ctypes.sizeof(TaskVmInfoRev1) != 152 or REV1_COUNT != 38:
        raise RuntimeError("unexpected task_vm_info rev1 ABI")
    if _lib is None:
        _lib = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
        _lib.task_info.argtypes = [
            ctypes.c_uint32,
            ctypes.c_int32,
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_uint32),
        ]
        _lib.task_info.restype = ctypes.c_int32
    task = ctypes.c_uint32.in_dll(_lib, "mach_task_self_").value
    value = TaskVmInfoRev1()
    count = ctypes.c_uint32(REV1_COUNT)
    code = _lib.task_info(
        task,
        TASK_VM_INFO,
        ctypes.cast(ctypes.byref(value), ctypes.POINTER(ctypes.c_int32)),
        ctypes.byref(count),
    )
    if code != 0:
        raise RuntimeError(f"task_info TASK_VM_INFO failed: kern_return_t={code}")
    if count.value != REV1_COUNT:
        raise RuntimeError(f"unsupported task_vm_info count={count.value}")
    if value.page_size <= 0 or value.page_size & (value.page_size - 1):
        raise RuntimeError("invalid Mach page size")
    return {
        "mach_resident_bytes": int(value.resident_size),
        "mach_resident_peak_bytes": int(value.resident_size_peak),
        "mach_internal_bytes": int(value.internal),
        "mach_external_bytes": int(value.external),
        "mach_compressed_bytes": int(value.compressed),
        "mach_compressed_peak_bytes": int(value.compressed_peak),
        "mach_phys_footprint_bytes": int(value.phys_footprint),
        "mach_virtual_bytes": int(value.virtual_size),
        "mach_page_size": int(value.page_size),
        "mach_region_count": int(value.region_count),
        "mach_task_info_count": int(count.value),
    }


def layout():
    return {
        "task_vm_info_flavor": TASK_VM_INFO,
        "natural_t_bytes": ctypes.sizeof(ctypes.c_uint32),
        "integer_t_bytes": ctypes.sizeof(ctypes.c_int32),
        "rev1_count": REV1_COUNT,
        "prefix_size": ctypes.sizeof(TaskVmInfoRev1),
        "offsets": {
            name: getattr(TaskVmInfoRev1, name).offset
            for name, _ in TaskVmInfoRev1._fields_
        },
    }
