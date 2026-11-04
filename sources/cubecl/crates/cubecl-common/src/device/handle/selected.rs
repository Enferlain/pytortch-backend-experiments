//! A transport chosen per device at runtime: the channel to a runner thread by
//! default, or the reentrant mutex, whose calls run on the calling thread.
//!
//! A caller that orders this runtime's work with its own on a shared stream (an
//! embedding framework that owns the streams) needs every call to have reached
//! the device's queue when it returns, which only the calling-thread transport
//! gives. The choice is made with [`set_transport`] before the device's first
//! service exists and is fixed from then on, so the rest of the process keeps
//! the channel.

use core::any::Any;
use std::sync::{Mutex, PoisonError};

use hashbrown::HashMap;

use super::channel::ChannelDeviceHandle;
use super::reentrant::ReentrantMutexDeviceHandle;
use crate::device::handle::{
    CallError, DeviceHandleSpec, ServerUtilitiesHandle, ServiceCloseError, ServiceCreationError,
    SpawnedCall,
};
use crate::device::{DeviceId, DeviceService};

/// How calls to a device's services reach them.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Transport {
    /// Calls are queued to the device's runner thread; non-blocking calls
    /// return before they run.
    #[default]
    Channel,
    /// Calls run on the calling thread under a reentrant lock, and have run
    /// when they return.
    CallingThread,
}

/// [`set_transport`] was called after the device's transport was fixed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TransportFixed {
    /// The device.
    pub device_id: DeviceId,
    /// The transport its services already use.
    pub transport: Transport,
}

impl core::fmt::Display for TransportFixed {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "device {:?} already uses the {:?} transport",
            self.device_id, self.transport
        )
    }
}

impl core::error::Error for TransportFixed {}

/// Each device's transport: chosen by [`set_transport`], or fixed to the
/// default by the first handle made for the device.
static TRANSPORTS: Mutex<Option<HashMap<DeviceId, Transport>>> = Mutex::new(None);

fn with_transports<R>(f: impl FnOnce(&mut HashMap<DeviceId, Transport>) -> R) -> R {
    let mut guard = TRANSPORTS.lock().unwrap_or_else(PoisonError::into_inner);
    f(guard.get_or_insert_with(HashMap::new))
}

/// Chooses the transport for every service on `device_id`.
///
/// # Errors
///
/// When a handle was already made for the device with another transport (the
/// default one, if no choice was made before): its services exist and stay
/// where they are.
pub fn set_transport(device_id: DeviceId, transport: Transport) -> Result<(), TransportFixed> {
    with_transports(|map| {
        let current = *map.entry(device_id).or_insert(transport);
        if current == transport {
            Ok(())
        } else {
            Err(TransportFixed {
                device_id,
                transport: current,
            })
        }
    })
}

/// The device's transport, fixing it to the default if none was chosen.
pub fn transport(device_id: DeviceId) -> Transport {
    with_transports(|map| *map.entry(device_id).or_default())
}

/// A handle through the transport chosen for its device.
#[derive(Clone)]
pub enum SelectedDeviceHandle {
    /// Through the runner thread.
    Channel(ChannelDeviceHandle),
    /// On the calling thread.
    CallingThread(ReentrantMutexDeviceHandle),
}

macro_rules! dispatch {
    ($self:expr, $handle:ident => $body:expr) => {
        match $self {
            SelectedDeviceHandle::Channel($handle) => $body,
            SelectedDeviceHandle::CallingThread($handle) => $body,
        }
    };
}

impl DeviceHandleSpec for SelectedDeviceHandle {
    /// Not every handle blocks; [`blocks`](DeviceHandleSpec::blocks) answers per handle.
    const BLOCKING: bool = false;

    fn blocks(&self) -> bool {
        dispatch!(self, handle => handle.blocks())
    }

    fn insert<S: DeviceService>(
        device_id: DeviceId,
        service: S,
    ) -> Result<Self, ServiceCreationError> {
        Ok(match transport(device_id) {
            Transport::Channel => Self::Channel(<ChannelDeviceHandle as DeviceHandleSpec>::insert(
                device_id, service,
            )?),
            Transport::CallingThread => Self::CallingThread(
                <ReentrantMutexDeviceHandle as DeviceHandleSpec>::insert(device_id, service)?,
            ),
        })
    }

    fn new<S: DeviceService>(device_id: DeviceId) -> Self {
        match transport(device_id) {
            Transport::Channel => Self::Channel(
                <ChannelDeviceHandle as DeviceHandleSpec>::new::<S>(device_id),
            ),
            Transport::CallingThread => Self::CallingThread(
                <ReentrantMutexDeviceHandle as DeviceHandleSpec>::new::<S>(device_id),
            ),
        }
    }

    fn device_id(&self) -> DeviceId {
        dispatch!(self, handle => handle.device_id())
    }

    fn utilities(&self) -> ServerUtilitiesHandle {
        dispatch!(self, handle => handle.utilities())
    }

    fn flush_queue(&self) {
        dispatch!(self, handle => handle.flush_queue())
    }

    fn submit_blocking<'a, R: Send, T: FnOnce(&mut dyn Any) -> R + Send + 'a>(
        &self,
        task: T,
    ) -> Result<R, CallError> {
        dispatch!(self, handle => handle.submit_blocking(task))
    }

    fn submit<T: FnOnce(&mut dyn Any) + Send + 'static>(&self, task: T) {
        dispatch!(self, handle => handle.submit(task))
    }

    fn spawn<R: Send + 'static, T: FnOnce(&mut dyn Any) -> R + Send + 'static>(
        &self,
        task: T,
    ) -> SpawnedCall<R> {
        dispatch!(self, handle => handle.spawn(task))
    }

    fn exclusive<R: Send, T: FnOnce() -> R + Send>(&self, task: T) -> Result<R, CallError> {
        dispatch!(self, handle => handle.exclusive(task))
    }

    fn shutdown(device_id: DeviceId) {
        match transport(device_id) {
            Transport::Channel => ChannelDeviceHandle::shutdown(device_id),
            Transport::CallingThread => ReentrantMutexDeviceHandle::shutdown(device_id),
        }
    }

    fn close<S: DeviceService>(device_id: DeviceId) -> Result<(), ServiceCloseError> {
        match transport(device_id) {
            Transport::Channel => ChannelDeviceHandle::close::<S>(device_id),
            Transport::CallingThread => ReentrantMutexDeviceHandle::close::<S>(device_id),
        }
    }
}
