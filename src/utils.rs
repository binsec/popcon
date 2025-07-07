/**************************************************************************/
/*  This file is part of POPCON.                                          */
/*                                                                        */
/*  Copyright (C) 2025                                                    */
/*    CEA (Commissariat à l'énergie atomique et aux énergies              */
/*         alternatives)                                                  */
/*                                                                        */
/*  you can redistribute it and/or modify it under the terms of the GNU   */
/*  Lesser General Public License as published by the Free Software       */
/*  Foundation, version 2.1.                                              */
/*                                                                        */
/*  It is distributed in the hope that it will be useful,                 */
/*  but WITHOUT ANY WARRANTY; without even the implied warranty of        */
/*  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the         */
/*  GNU Lesser General Public License for more details.                   */
/*                                                                        */
/*  See the GNU Lesser General Public License version 2.1                 */
/*  for more details (enclosed in the file licenses/LGPLv2.1).            */
/*                                                                        */
/**************************************************************************/

//! Misc utils.

use anyhow::Context;
use nix::fcntl::{fcntl, FcntlArg, FdFlag, OFlag};
use num_bigint::BigUint;
use num_traits::Zero;
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{Read, Write};
use std::ops::{Index, IndexMut};
use std::os::unix::io::AsRawFd;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tracing::trace;

/// Sets or clears the cloexec flag on this file descriptor
pub fn set_cloexec<F: AsRawFd>(f: &F, value: bool) -> anyhow::Result<()> {
    let fd = f.as_raw_fd();
    let flags = fcntl(fd, FcntlArg::F_GETFD).context("fnctl(getfd)")?;
    // safety: on the result of fcntl
    let mut flags = unsafe { FdFlag::from_bits_unchecked(flags) };
    flags.set(FdFlag::FD_CLOEXEC, value);
    fcntl(fd, FcntlArg::F_SETFD(flags)).context("fnctl(setfd)")?;
    Ok(())
}

/// Sets or clears the blocking flag on this file descriptor
pub fn set_blocking<F: AsRawFd>(f: &F, value: bool) -> anyhow::Result<()> {
    let fd = f.as_raw_fd();
    let flags = fcntl(fd, FcntlArg::F_GETFL).context("fnctl(getfl)")?;
    // safety: on the result of fcntl
    let mut flags = unsafe { OFlag::from_bits_unchecked(flags) };
    flags.set(OFlag::O_NONBLOCK, value);
    fcntl(fd, FcntlArg::F_SETFL(flags)).context("fnctl(setfl)")?;
    Ok(())
}

/// A map where all unset values have value 0.
/// # Example
/// ```
/// use popcon::utils::CountingMap;
/// use num_bigint::BigUint;
///
/// let mut foo = CountingMap::new();
/// assert_eq!(foo[&1], 0u32.into());
/// foo[&2] += BigUint::from(2u32);
/// foo[&3] = 3u32.into();
/// assert_eq!(foo.max(), 3u32.into());
/// ```
pub struct CountingMap<T: Ord> {
    /// The actual map
    map: BTreeMap<T, BigUint>,
    /// just to be able to return a reference to missing values in Index
    zero: BigUint,
}

impl<T: Ord + std::fmt::Debug> std::fmt::Debug for CountingMap<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", &self.map)
    }
}

impl<T: Ord> PartialEq for CountingMap<T> {
    fn eq(&self, other: &Self) -> bool {
        for (key, value) in self.map.iter() {
            if value != &other[key] {
                return false;
            }
        }
        for (key, value) in other.map.iter() {
            if value != &self[key] {
                return false;
            }
        }
        true
    }
}

impl<T: Ord> CountingMap<T> {
    /// creates an empty map, where all keys are associated with 0
    pub fn new() -> Self {
        Self {
            map: BTreeMap::new(),
            zero: BigUint::zero(),
        }
    }

    /// returns the max value of the map
    pub fn max(&self) -> BigUint {
        let max: Option<&BigUint> = self.map.values().max();
        max.cloned().unwrap_or_else(BigUint::zero)
    }

    /// returns the max value of the map, and if the map is not empty, a value reaching it.
    pub fn argmax(&self) -> (Option<&T>, &BigUint) {
        let mut res = None;
        let mut count = &self.zero;
        for (k, v) in self.map.iter() {
            if v > count {
                count = v;
                res = Some(k);
            }
        }
        (res, count)
    }

    /// Returns the number of non-zero entries in the counting map
    pub fn count_non_zero(&self) -> usize {
        self.map.values().filter(|x| !x.is_zero()).count()
    }
}

impl<T: Ord> Index<&T> for CountingMap<T> {
    type Output = BigUint;
    fn index(&self, key: &T) -> &Self::Output {
        &self.map.get(key).unwrap_or(&self.zero)
    }
}

impl<T: Ord + Clone> IndexMut<&T> for CountingMap<T> {
    fn index_mut(&mut self, key: &T) -> &mut Self::Output {
        let e = self.map.entry(key.clone());
        e.or_insert_with(BigUint::zero)
    }
}

const SIZE: usize = 1024;

/// A buffer that contains at least the 512 last bytes read.
struct RingBuffer {
    data: [u8; SIZE],
    end: usize,
    error: std::io::Result<()>,
    done: bool,
}

impl RingBuffer {
    /// you shouldn't read from a dead RingBuffer
    fn dead(&self) -> bool {
        self.done || self.error.is_err()
    }

    /// Register that read(buffer) returned res
    fn consume(&mut self, buffer: &[u8], res: std::io::Result<usize>) {
        let half = SIZE / 2;
        if self.end > half {
            self.data.copy_within((self.end - half)..self.end, 0);
            self.end = self.end - half;
        }
        match res {
            Ok(0) => self.done = true,
            Ok(n) => {
                self.data[self.end..(self.end + n)].copy_from_slice(&buffer[0..n]);
                self.end += n;
            }
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
            Err(e) => self.error = Err(e),
        }
    }

    /// Get the last 512 to 1024 bytes read, and possibly an error.
    fn result(&self) -> (&[u8], std::io::Result<()>) {
        let text = &self.data[..self.end];
        let error = match &self.error {
            Ok(()) => Ok(()),
            Err(e) => Err(std::io::Error::from(e.kind())),
        };
        (text, error)
    }

    /// Creates an empty buffer
    fn new() -> Self {
        Self {
            data: [0; SIZE],
            end: 0,
            error: Ok(()),
            done: false,
        }
    }
}

/// Something to consume the reading end of a pipe to prevent the other end
/// from blocking. Only the 512 to 1024 last bytes and first error encountered are
/// kept.
/// Actually spawns a thread, which should quit when this is dropped.
pub struct LastLines {
    buf: Arc<Mutex<RingBuffer>>,
}

impl LastLines {
    /// Spawns a thread consuming this file descriptor, and returns a handle to get
    /// the end.
    pub fn new<R: Read + AsRawFd + Send + 'static>(mut read: R) -> Self {
        let buf = Arc::new(Mutex::new(RingBuffer::new()));
        let res = Self { buf: buf.clone() };
        std::thread::spawn(move || {
            let mut buffer = [0; SIZE / 2];
            loop {
                let res = read.read(&mut buffer);
                let mut inner = buf.lock().unwrap();
                inner.consume(&buffer, res);
                if inner.dead() {
                    break;
                }
            }
            drop(read);
        });
        res
    }

    /// peek at the currently last bytes read.
    pub fn get(&self) -> (Vec<u8>, std::io::Result<()>) {
        let inner = self.buf.lock().unwrap();
        let (text, err) = inner.result();
        (text.to_owned(), err)
    }
}

impl Drop for LastLines {
    fn drop(&mut self) {
        if let Ok(mut inner) = self.buf.lock() {
            inner.done = true;
            drop(inner);
        }
    }
}

#[test]
fn last_lines() {
    use std::fs::File;
    use std::io::Write;
    use std::os::unix::io::FromRawFd;
    let (raw_read, raw_write) =
        nix::unistd::pipe().expect("creating a pipe for bitblasting output");
    let (read, mut write) = unsafe { (File::from_raw_fd(raw_read), File::from_raw_fd(raw_write)) };
    let lines = LastLines::new(read);
    for i in 0..=1_000_000 {
        writeln!(&mut write, "{}", i).unwrap();
    }
    std::thread::sleep(std::time::Duration::from_secs(2));
    let (txt, err) = lines.get();
    assert!(err.is_ok());
    let expected = b"1000000\n";
    assert_eq!(&txt[(txt.len() - expected.len())..txt.len()], expected);
}

/// Wait for a process for at most the specified duration
pub fn try_wait_timeout(
    process: &mut std::process::Child,
    timeout: Duration,
) -> std::io::Result<Option<std::process::ExitStatus>> {
    let mut waited = Duration::from_secs(0);
    let interval = Duration::from_millis(50);
    while waited < timeout {
        match process.try_wait()? {
            Some(e) => return Ok(Some(e)),
            None => (),
        }
        std::thread::sleep(interval);
        waited += interval;
    }
    Ok(None)
}

/// Returns the 2-based logarithm of biguint at 0.1 precision.
/// # Example
/// ```
/// pub use popcon::utils::log2;
/// pub use num_bigint::BigUint;
/// assert_eq!(log2(&0u32.into()), -f32::INFINITY);
/// assert_eq!(log2(&2u32.into()), 1.0_f32);
/// assert!((log2(&3u32.into()) - 3.0_f32.log2()).abs() <= 0.1);
/// ```
pub fn log2(n: &BigUint) -> f32 {
    if n.is_zero() {
        return f32::NEG_INFINITY;
    } else {
        ((n.pow(10).bits() - 1) as f64 / 10.) as f32
    }
}

/// Wraps a writer. All writes directed to this writer are repeated to the wrapped
/// writer, and if the loglevel is TRACE, to a logfile whose path is logged.
pub struct LoggingWriter<W: Write> {
    /// The logging file where to log writes, or None if the loglevel is too high
    logfile: Option<File>,
    /// The wrapped writer
    writer: W,
}

impl<W: Write> Write for LoggingWriter<W> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let n = self.writer.write(buf)?;
        if let Some(ref mut logfile) = &mut self.logfile {
            logfile.write_all(&buf[..n])?;
        }
        Ok(n)
    }
    fn flush(&mut self) -> std::io::Result<()> {
        let res = match &mut self.logfile {
            Some(ref mut l) => l.flush(),
            None => Ok(()),
        };
        self.writer.flush()?;
        res
    }
}

/// Returns whether a message with this level would be printed
pub fn would_log(level: tracing::Level) -> bool {
    level < tracing::level_filters::LevelFilter::current()
}

impl<W: Write> LoggingWriter<W> {
    /// Creates a new LoggingWriter. Writing to this writer writes to `w`, and if the logging level
    /// is TRACE then also to a temporary file whose path is logged.
    pub fn new(w: W) -> anyhow::Result<LoggingWriter<W>> {
        let logfile = if would_log(tracing::Level::DEBUG) {
            let (file, path) = tempfile::NamedTempFile::new()?.keep()?;
            trace!(path =%path.display(), "logging output to a file");
            Some(file)
        } else {
            None
        };
        Ok(LoggingWriter { logfile, writer: w })
    }
}

/// A temporary file that is removed on drop only if debugging is not enabled.
pub enum MaybePersistentTempFile {
    /// The file will be removed
    Temp(tempfile::NamedTempFile),
    /// The file is Persistent: file descriptor and path
    Persistent(std::fs::File, std::path::PathBuf),
}

impl MaybePersistentTempFile {
    /// Creates a temporary file, which will only be cleaned up if debugging is not enabled.
    /// Extension can be chosen (including leading dot).
    pub fn new(extension: &str) -> anyhow::Result<Self> {
        let temp = tempfile::Builder::new().suffix(extension).tempfile()?;
        if would_log(tracing::Level::DEBUG) {
            let (file, path) = temp.keep()?;
            trace!("Persisting temporary data to {}", path.display());
            Ok(Self::Persistent(file, path))
        } else {
            Ok(Self::Temp(temp))
        }
    }

    /// Returns the underlying File.
    pub fn as_file_mut(&mut self) -> &mut File {
        match self {
            MaybePersistentTempFile::Temp(file) => file.as_file_mut(),
            MaybePersistentTempFile::Persistent(file, _) => file,
        }
    }
}

impl Write for MaybePersistentTempFile {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.as_file_mut().write(buf)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.as_file_mut().flush()
    }
}

impl AsRef<Path> for MaybePersistentTempFile {
    fn as_ref(&self) -> &Path {
        match self {
            MaybePersistentTempFile::Persistent(_, path) => path.as_ref(),
            MaybePersistentTempFile::Temp(file) => file.as_ref(),
        }
    }
}

/// Represents a stream of bytes that can be read one or several times.
pub trait Input: Copy {
    /// type for to_read
    type R: Read;
    /// type for to_path
    type F: AsRef<Path>;
    // /// type to into_path
    // type F2: AsRef<Path>;
    /// how displayed
    type D: std::fmt::Display;
    /// Returns a path that can be read several times, as long as the result is not dropped.
    fn to_path(&self) -> anyhow::Result<Self::F>;
    // /// Returns a path that can be read only once, and not after the result is dropped.
    // fn into_path(&self) -> anyhow::Result<Self::F2>;
    /// Returns a readable object that reads the data, can be called several times.
    fn to_read(&self) -> anyhow::Result<Self::R>;
    /// Returns the owned bytes of the data.
    fn into_bytes(&self) -> anyhow::Result<Vec<u8>>;
    /// A debug string to pritn
    fn display(&self) -> Self::D;
}

impl<'a> Input for &'a [u8] {
    type R = &'a [u8];
    type F = MaybePersistentTempFile;
    // type F2 = tempfile::NamedTempFile;
    type D = &'static str;
    fn to_path(&self) -> anyhow::Result<MaybePersistentTempFile> {
        let mut file =
            MaybePersistentTempFile::new("").context("creating temp file to dump input")?;
        file.as_file_mut()
            .write_all(self)
            .context("dumping input string to temp file")?;
        Ok(file)
    }

    // fn into_path(&self) -> anyhow::Result<tempfile::NamedTempFile> {
    //     self.to_path()
    // }

    fn to_read(&self) -> anyhow::Result<&'a [u8]> {
        Ok(self)
    }

    fn into_bytes(&self) -> anyhow::Result<Vec<u8>> {
        Ok((*self).to_owned())
    }

    fn display(&self) -> &'static str {
        "<in memory>"
    }
}

impl<'a> Input for &'a Path {
    type R = File;
    type F = &'a Path;
    // type F2 = &'a Path;
    type D = std::path::Display<'a>;
    fn to_path(&self) -> anyhow::Result<&'a Path> {
        Ok(self)
    }

    // fn into_path(&self) -> anyhow::Result<&'a Path> {
    //     Ok(self)
    // }

    fn to_read(&self) -> anyhow::Result<File> {
        File::open(self)
            .with_context(|| format!("opening input file {} for reading", self.display()))
    }

    fn into_bytes(&self) -> anyhow::Result<Vec<u8>> {
        let mut file = self.to_read()?;
        let mut res = Vec::new();
        file.read_to_end(&mut res)
            .with_context(|| format!("reading input file {}", self.display()))?;
        Ok(res)
    }

    fn display(&self) -> Self::D {
        Path::display(self)
    }
}
