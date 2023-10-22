package com.Fintech.OnlineBanking.controller;
import org.springframework.beans.factory.annotation.Autowired;

import org.springframework.http.ResponseEntity;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;
import com.Fintech.OnlineBanking.domain.Message;
import com.Fintech.OnlineBanking.dto.CreateRequestCardRequest;
import com.Fintech.OnlineBanking.service.RequestCardService;
import com.Fintech.OnlineBanking.user.User;

@RestController
@RequestMapping("/create_Request_card")
public class CreateRequestCardController {
	@Autowired
	private RequestCardService requestCardService;
	
	@PostMapping("/fill_information")
	public ResponseEntity<?> registerFillInformation(@AuthenticationPrincipal User user,@RequestBody CreateRequestCardRequest createRequestCardRequest)
	{
		     Message message = requestCardService.saveRequestCreateVisa(user,createRequestCardRequest);
				
		     return ResponseEntity.ok(message);
	}
	


}
