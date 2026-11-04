// Transport choice per device: the calling-thread transport has run a
// non-blocking call when it returns; a device with no choice keeps the channel.

fn calling_thread(device_id: DeviceId) -> DeviceHandle<TestDeviceState<1>> {
    super::set_transport(device_id, super::Transport::CallingThread).unwrap();
    DeviceHandle::<TestDeviceState<1>>::new(device_id)
}

#[test]
fn test_calling_thread_runs_submit_before_returning() {
    let handle = DeviceFixture::new(calling_thread, DeviceHandle::<TestDeviceState<1>>::shutdown);
    assert!(handle.blocks());
    let ran_on = Arc::new(std::sync::Mutex::new(None));
    let record = ran_on.clone();
    handle.submit(move |state| {
        state.counter += 1;
        *record.lock().unwrap() = Some(std::thread::current().id());
    });
    // No flush and no blocking call: the task has already run, here.
    assert_eq!(*ran_on.lock().unwrap(), Some(std::thread::current().id()));
    assert_eq!(handle.submit_blocking(|state| state.counter).unwrap(), 1);
}

#[test]
fn test_default_device_keeps_the_channel() {
    let handle = DeviceFixture::new(
        DeviceHandle::<TestDeviceState<1>>::new,
        DeviceHandle::<TestDeviceState<1>>::shutdown,
    );
    assert!(!handle.blocks());
    let runner = handle
        .submit_blocking(|_| std::thread::current().id())
        .unwrap();
    assert_ne!(runner, std::thread::current().id());
}

#[test]
fn test_transport_is_fixed_once_used() {
    let handle = DeviceFixture::new(
        DeviceHandle::<TestDeviceState<1>>::new,
        DeviceHandle::<TestDeviceState<1>>::shutdown,
    );
    let device_id = handle.device_id();
    let error = super::set_transport(device_id, super::Transport::CallingThread).unwrap_err();
    assert_eq!(error.transport, super::Transport::Channel);
    // Asking again for the transport in use is not an error.
    super::set_transport(device_id, super::Transport::Channel).unwrap();

    let blocking = DeviceFixture::new(calling_thread, DeviceHandle::<TestDeviceState<1>>::shutdown);
    let error =
        super::set_transport(blocking.device_id(), super::Transport::Channel).unwrap_err();
    assert_eq!(error.transport, super::Transport::CallingThread);
}
