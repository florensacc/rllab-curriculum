all: hopper.xml cartpole.xml double_pendulum.xml half_cheetah.xml

hopper.xml: gen_xml.rb hopper.xml.rb
	ruby gen_xml.rb hopper.xml.rb > hopper.xml
	
half_cheetah.xml: gen_xml.rb half_cheetah.xml.rb
	ruby gen_xml.rb half_cheetah.xml.rb > half_cheetah.xml

cartpole.xml: gen_xml.rb cartpole.xml.rb
	ruby gen_xml.rb cartpole.xml.rb > cartpole.xml

double_pendulum.xml: gen_xml.rb double_pendulum.xml.rb
	ruby gen_xml.rb double_pendulum.xml.rb > double_pendulum.xml

test_cartpole:
	python cartpole.py

clean:
	rm *.xml
