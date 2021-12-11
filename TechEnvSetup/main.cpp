#include <wic/camerafinder.h>
#include <wic/framegrabber.h>
#include <wic/wic.h>

#include <iostream>
#include <vector>

int
main()
{
	auto serialNumber = "";
	auto wic = wic::findAndConnect(serialNumber);

	if (!wic) {
		std::cerr << "Could not connect WIC: " << serialNumber << std::endl;
		return 1;
	}

	auto defaultRes = wic->doDefaultWICSettings();
	if (defaultRes.first != wic::ResponseStatus::Ok) {
		std::cerr << "DoDefaultWICSettings: "
				  << wic::responseStatusToStr(defaultRes.first) << std::endl;
		return 2;
	}

	// enable advanced features
	wic->iKnowWhatImDoing();
	// set advanced radiometry if core supports it
	auto ar = wic->setAR(wic::AR::True);
	if (ar.first == wic::ResponseStatus::Ok) {
		// set core temperature resolution, value depends on core calibration
		wic->setTempRes(wic::ARRes::High);
	}
	// set core gain
	auto gain = wic->setGain(wic::GainMode::High);

	auto grabber = wic->frameGrabber();
	grabber->setup();

	auto resolution = wic->getResolution();
	if (resolution.first == 0 || resolution.second == 0) {
		std::cerr << "Invalid resolution, core detection error." << std::endl;
		return 3;
	}

	// default wic settings = OutputType::RAD14
	// every 2 bytes represent radiometric flux of one pixel
	// buffer is in row major format
	std::vector<uint8_t> buffer;
	buffer.reserve(resolution.first * resolution.second * sizeof(uint16_t));

	while (true) {
		grabber->getBuffer(buffer, 1000);
		//...
	}
}
